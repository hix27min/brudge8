#!/usr/bin/env python3
"""
Nekto.me Audio Bridge — HOSTED version.

Отличия от десктопной версии:
- Нет sounddevice (на сервере нет звуковой карты)
- Твой браузер (телефон/ПК) подключается к серверу через WebRTC:
    - шлёт свой микрофон на сервер
    - получает обратно смешанный поток (голос A + голос B)
- Сервер маршрутизирует голоса между A, B и твоим браузером.

Архитектура:
    Собеседник A <──nekto──> [server: aiortc]
                                   │ mix
    Собеседник B <──nekto──> [server: aiortc]
                                   │
    Твой телефон <──WebRTC──> [server: HTTPS + aiortc]
                (микрофон)             (голоса A+B тебе)
"""
import asyncio
import base64
import fractions
import hashlib
import json
import logging
import os
import sys
import time

import aiohttp
import numpy as np
import av
from aiohttp import web
from aiortc import (
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.contrib.media import MediaRelay
from aiortc.mediastreams import MediaStreamTrack
from aiortc.sdp import candidate_from_sdp
from av.audio.frame import AudioFrame

# ============================================================
# КОНСТАНТЫ
# ============================================================
SAMPLE_RATE = 48000
FRAME_SIZE = 960                    # 20 ms на 48 кГц
CHANNELS = 1

NEKTO_WS = "wss://audio.nekto.me/websocket/?EIO=3&transport=websocket"
NEKTO_ORIGIN = "https://nekto.me"

WEBAGENT_SALT_1 = "BYdKPTYYGZ7ALwA"
WEBAGENT_SALT_2 = "8oNm2"

log = logging.getLogger("nekto")


def compute_web_agent(user_id: str, internal_id) -> str:
    s = f"{user_id}{WEBAGENT_SALT_1}{WEBAGENT_SALT_2}{internal_id}"
    return base64.b64encode(
        hashlib.sha256(s.encode()).hexdigest().encode()
    ).decode()


# ============================================================
# КАНАЛ — кольцевой буфер сэмплов
# ============================================================
class SourceChannel:
    """
    Канал с кольцевым буфером. Два режима:
    1. track != None — сам запускает _pull(), читает через AudioResampler
    2. track == None — внешний код вызывает push_samples()
    """
    def __init__(self, track, name="", use_resampler=True, preroll_ms=60):
        self.track = track
        self.name = name
        self._buf = np.zeros(SAMPLE_RATE * 2, dtype=np.int16)
        self._write_pos = 0
        self._read_pos = 0
        self._available = 0
        self.alive = True
        self._prerolled = preroll_ms == 0
        self._preroll_samples = int(SAMPLE_RATE * preroll_ms / 1000)
        self._resampler = av.AudioResampler(
            format="s16", layout="mono", rate=SAMPLE_RATE,
        ) if use_resampler else None
        self.task = None
        if track is not None:
            self.task = asyncio.create_task(self._pull())

    async def _pull(self):
        try:
            while self.alive:
                frame = await self.track.recv()
                if self._resampler is not None:
                    try:
                        out_frames = self._resampler.resample(frame)
                    except Exception as e:
                        log.debug(f"{self.name} resample: {e}")
                        continue
                    for f in out_frames:
                        arr = f.to_ndarray().flatten().astype(np.int16)
                        self._write_samples(arr)
                else:
                    arr = frame.to_ndarray().flatten().astype(np.int16)
                    self._write_samples(arr)
        except Exception as e:
            log.debug(f"source {self.name} ended: {e}")

    def push_samples(self, arr):
        self._write_samples(arr)

    def _write_samples(self, arr):
        buflen = len(self._buf)
        n = len(arr)
        if self._available + n > buflen:
            drop = self._available + n - buflen
            self._read_pos = (self._read_pos + drop) % buflen
            self._available -= drop
        end = self._write_pos + n
        if end <= buflen:
            self._buf[self._write_pos:end] = arr
        else:
            first = buflen - self._write_pos
            self._buf[self._write_pos:] = arr[:first]
            self._buf[:n - first] = arr[first:]
        self._write_pos = (self._write_pos + n) % buflen
        self._available += n

    def read_samples(self, n):
        if not self._prerolled:
            if self._available >= self._preroll_samples:
                self._prerolled = True
            else:
                return np.zeros(n, dtype=np.int16)
        if self._available < n:
            out = np.zeros(n, dtype=np.int16)
            k = self._available
            if k > 0:
                buflen = len(self._buf)
                end = self._read_pos + k
                if end <= buflen:
                    out[:k] = self._buf[self._read_pos:end]
                else:
                    first = buflen - self._read_pos
                    out[:first] = self._buf[self._read_pos:]
                    out[first:k] = self._buf[:k - first]
                self._read_pos = (self._read_pos + k) % buflen
                self._available = 0
            self._prerolled = False
            return out
        buflen = len(self._buf)
        out = np.empty(n, dtype=np.int16)
        end = self._read_pos + n
        if end <= buflen:
            out[:] = self._buf[self._read_pos:end]
        else:
            first = buflen - self._read_pos
            out[:first] = self._buf[self._read_pos:]
            out[first:] = self._buf[:n - first]
        self._read_pos = (self._read_pos + n) % buflen
        self._available -= n
        return out

    def stop(self):
        self.alive = False
        if self.task:
            self.task.cancel()


# ============================================================
# МИКШЕР
# ============================================================
class MixerTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, name="mixer"):
        super().__init__()
        self.name = name
        self.sources: dict = {}
        self._pts = 0
        self._start = None
        self.source_rms: dict = {}

    def add_source_channel(self, name, channel, gain_fn):
        self.remove_source(name)
        self.sources[name] = (channel, gain_fn)

    def remove_source(self, name):
        if name in self.sources:
            ch, _ = self.sources.pop(name)
            ch.stop()
        self.source_rms.pop(name, None)

    async def recv(self):
        if self._start is None:
            self._start = time.time()
        target = self._start + self._pts / SAMPLE_RATE
        delay = target - time.time()
        if delay > 0:
            await asyncio.sleep(delay)
        elif delay < -0.1:
            self._start = time.time() - self._pts / SAMPLE_RATE

        mixed = np.zeros(FRAME_SIZE, dtype=np.int32)
        for nm, (ch, gf) in list(self.sources.items()):
            g = gf()
            arr = ch.read_samples(FRAME_SIZE)
            rms = float(np.sqrt(np.mean(arr.astype(np.float32) ** 2)))
            prev = self.source_rms.get(nm, 0.0)
            self.source_rms[nm] = prev * 0.5 + rms * 0.5
            if g <= 0:
                continue
            if g == 1.0:
                mixed += arr.astype(np.int32)
            else:
                mixed += (arr.astype(np.float32) * g).astype(np.int32)

        mixed = np.clip(mixed, -32768, 32767).astype(np.int16)
        frame = AudioFrame.from_ndarray(mixed.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = SAMPLE_RATE
        frame.pts = self._pts
        frame.time_base = fractions.Fraction(1, SAMPLE_RATE)
        self._pts += FRAME_SIZE
        return frame


# ============================================================
# BROWSER MIC SOURCE (WebSocket-based)
# Принимает raw PCM от браузера и раздаёт push-каналам (A, B).
# ============================================================
class BrowserMicSource:
    """
    Принимает PCM-аудио от браузера через WebSocket и раздаёт
    одни и те же сэмплы трём push-каналам (A, B, self-monitor).

    Формат входных данных: int16 mono @ 16kHz (маленький, чтобы пролезло
    через TCP без лагов). Ресемплируем в 48kHz для Nekto.
    """
    def __init__(self):
        self.subscribers: list = []
        self.current_rms = 0.0
        # ресемплер 16k → 48k
        self._resampler = av.AudioResampler(
            format="s16", layout="mono", rate=SAMPLE_RATE,
        )

    def subscribe(self, source_channel):
        self.subscribers.append(source_channel)

    def push_pcm(self, pcm_bytes, input_rate=16000):
        """Принимаем PCM-байты (int16 mono), ресемплируем и раздаём."""
        try:
            arr_16k = np.frombuffer(pcm_bytes, dtype=np.int16)
            if len(arr_16k) == 0:
                return
            # создаём AudioFrame и ресемплим в 48k mono
            frame = AudioFrame.from_ndarray(
                arr_16k.reshape(1, -1), format="s16", layout="mono"
            )
            frame.sample_rate = input_rate
            frame.pts = None
            out_frames = self._resampler.resample(frame)
            for f in out_frames:
                arr_48k = f.to_ndarray().flatten().astype(np.int16)
                # RMS для VU
                rms = float(np.sqrt(np.mean(arr_48k.astype(np.float32) ** 2)))
                self.current_rms = self.current_rms * 0.5 + rms * 0.5
                for sub in self.subscribers:
                    try:
                        sub.push_samples(arr_48k)
                    except Exception:
                        pass
        except Exception as e:
            log.debug(f"push_pcm: {e}")


# ============================================================
# NEKTO CLIENT
# ============================================================
class NektoClient:
    def __init__(self, name, outgoing_mixer, relay, on_remote_track, broadcast):
        self.name = name
        self.outgoing_mixer = outgoing_mixer
        self.relay = relay
        self.on_remote_track = on_remote_track
        self.broadcast = broadcast

        self.user_id = None
        self.user_agent = None
        self.locale = "ru"
        self.time_zone = "Europe/Moscow"
        self.search_criteria = None

        self.ws = None
        self.pc = None
        self.connection_id = None
        self.internal_id = None
        self.state = "idle"
        self.session = None
        self._ping_task = None
        self._read_task = None
        self._should_stay = False
        self._reconn = 0
        self._max_reconn = 5

    def _log(self, msg, lvl="info"):
        getattr(log, lvl, log.info)(f"[{self.name}] {msg}")
        self.broadcast("log", {"side": self.name.lower(), "msg": msg, "level": lvl})

    def _set_state(self, s):
        self.state = s
        self.broadcast("state", {"side": self.name.lower(), "state": s})

    @property
    def is_firefox(self) -> bool:
        return bool(self.user_agent) and "Gecko" in self.user_agent

    async def connect(self, user_id, user_agent, criteria, session):
        if self.ws and not self.ws.closed:
            self._log("already connected", "warning")
            return
        if not user_id:
            self._log("authToken не задан", "error"); return
        if not user_agent:
            self._log("user-agent не задан", "error"); return
        self.user_id = user_id
        self.user_agent = user_agent
        self.search_criteria = criteria
        self.session = session
        self._should_stay = True
        self._reconn = 0
        self._set_state("connecting")
        await self._do_connect()

    async def _do_connect(self):
        self._log(f"connecting (try {self._reconn + 1}/{self._max_reconn})")
        try:
            self.ws = await self.session.ws_connect(
                NEKTO_WS,
                headers={"Origin": NEKTO_ORIGIN, "User-Agent": self.user_agent},
                heartbeat=None,
            )
            self._reconn = 0
            self._log("WS opened", "info")
            self._read_task = asyncio.create_task(self._read_loop())
        except Exception as e:
            self._log(f"WS error: {e}", "error")
            await self._schedule_reconnect()

    async def _schedule_reconnect(self):
        if not self._should_stay:
            self._set_state("idle"); return
        if self._reconn >= self._max_reconn:
            self._log("max reconnect attempts", "error")
            self._set_state("idle"); return
        self._reconn += 1
        delay = 2 * (2 ** (self._reconn - 1))
        self._log(f"reconnecting in {delay}s...", "warning")
        await asyncio.sleep(delay)
        if self._should_stay:
            await self._do_connect()

    async def _read_loop(self):
        try:
            async for msg in self.ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    await self._on_ws_message(msg.data)
                elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR,
                                  aiohttp.WSMsgType.CLOSED):
                    break
        except Exception as e:
            self._log(f"read loop: {e}", "warning")
        finally:
            if self._ping_task:
                self._ping_task.cancel(); self._ping_task = None
            self._log("WS closed", "warning")
            await self._close_peer()
            await self._schedule_reconnect()

    async def _on_ws_message(self, raw):
        if not raw: return
        t = raw[0]
        if t == "0":
            p = json.loads(raw[1:])
            self._log(f"handshake sid={p['sid'][:8]}...")
            self._ping_task = asyncio.create_task(self._ping_loop(p["pingInterval"] / 1000))
            await self._send("40")
        elif t == "3":
            pass
        elif t == "4":
            sub = raw[1:2]
            if sub == "0":
                self._log("socket.io connected → register", "info")
                await self._send_register()
            elif sub == "2":
                arr = json.loads(raw[2:])
                if len(arr) >= 2 and arr[0] == "event":
                    await self._on_event(arr[1])

    async def _ping_loop(self, interval):
        try:
            while True:
                await asyncio.sleep(interval)
                await self._send("2")
        except asyncio.CancelledError:
            pass

    async def _send(self, raw):
        if self.ws and not self.ws.closed:
            try: await self.ws.send_str(raw)
            except Exception as e: log.debug(f"send: {e}")

    async def _emit_event(self, obj):
        await self._send("42" + json.dumps(["event", obj]))

    async def _send_register(self):
        payload = {
            "type": "register", "android": False, "version": 23,
            "userId": self.user_id, "timeZone": self.time_zone, "locale": self.locale,
        }
        if self.is_firefox:
            payload["firefox"] = True
        await self._emit_event(payload)
        self._log(f"sent register (userId={self.user_id[:8]}..., firefox={self.is_firefox})")

    async def _send_web_agent(self):
        if self.internal_id is None:
            self._log("no internal_id", "warning"); return
        data = compute_web_agent(self.user_id, self.internal_id)
        await self._emit_event({"type": "web-agent", "data": data})
        self._log("sent web-agent")

    async def _search(self):
        await self._emit_event({
            "type": "scan-for-peer", "peerToPeer": True, "token": None,
            "searchCriteria": self.search_criteria,
        })

    async def _on_event(self, d):
        t = d.get("type")
        if t == "registered":
            self.internal_id = d.get("internal_id")
            self._log(f"registered (internal_id={self.internal_id})", "info")
            await self._send_web_agent()
            self._set_state("searching")
            await self._search()
        elif t == "search.success":
            self._log("search active")
        elif t == "peer-connect":
            await self._on_peer_connect(d)
        elif t == "offer":
            await self._on_remote_offer(d)
        elif t == "answer":
            await self._on_remote_answer(d)
        elif t == "ice-candidate":
            await self._on_remote_ice(d)
        elif t == "peer-disconnect":
            await self._on_peer_disconnect(d)
        elif t == "peer-mute":
            self._log(f"peer {'muted' if d.get('muted') else 'unmuted'}")
        elif t == "ban":
            self._log(f"BAN: {d.get('banInfo')}", "error")
            self._should_stay = False
        elif t == "error":
            self._log(f"error #{d.get('id')}: {d.get('description')}", "error")

    async def _on_peer_connect(self, d):
        self.connection_id = d["connectionId"]
        turn = json.loads(d["turnParams"])
        turn_filtered = [t for t in turn if not t["url"].startswith("turn:[")]
        ice_servers = [
            RTCIceServer(urls=t["url"], username=t["username"], credential=t["credential"])
            for t in turn_filtered
        ]
        if d.get("stunUrl"):
            ice_servers.insert(0, RTCIceServer(urls=d["stunUrl"]))
        self._log(f"peer found (initiator={d['initiator']})", "info")
        self._set_state("peer-found")

        self.pc = RTCPeerConnection(RTCConfiguration(iceServers=ice_servers))

        @self.pc.on("track")
        async def on_track(track):
            self._log(f"got remote {track.kind} track", "info")
            if track.kind == "audio" and self.on_remote_track:
                self.on_remote_track(track)
            await self._emit_event({
                "type": "stream-received", "connectionId": self.connection_id,
            })

        @self.pc.on("connectionstatechange")
        async def on_state():
            if not self.pc: return
            cs = self.pc.connectionState
            self._log(f"connection state: {cs}")
            if cs == "connected":
                self._set_state("talking")
                try:
                    await self._emit_event({
                        "type": "peer-connection", "connection": True,
                        "connectionId": self.connection_id,
                    })
                except Exception:
                    pass

        if d["initiator"]:
            track_for_pc = self.relay.subscribe(self.outgoing_mixer)
            self.pc.addTrack(track_for_pc)
            offer = await self.pc.createOffer()
            await self.pc.setLocalDescription(offer)
            await self._emit_event({
                "type": "peer-mute", "connectionId": self.connection_id, "muted": False,
            })
            await self._emit_event({
                "type": "offer", "connectionId": self.connection_id,
                "offer": json.dumps({
                    "type": self.pc.localDescription.type,
                    "sdp": self.pc.localDescription.sdp,
                }),
            })
            self._log("offer sent")

    async def _on_remote_offer(self, d):
        offer = json.loads(d["offer"])
        await self.pc.setRemoteDescription(
            RTCSessionDescription(sdp=offer["sdp"], type="offer")
        )
        track_for_pc = self.relay.subscribe(self.outgoing_mixer)
        self.pc.addTrack(track_for_pc)
        ans = await self.pc.createAnswer()
        await self.pc.setLocalDescription(ans)
        await self._emit_event({
            "type": "answer", "connectionId": self.connection_id,
            "answer": json.dumps({
                "type": self.pc.localDescription.type,
                "sdp": self.pc.localDescription.sdp,
            }),
        })
        self._log("answer sent")

    async def _on_remote_answer(self, d):
        ans = json.loads(d["answer"])
        await self.pc.setRemoteDescription(
            RTCSessionDescription(sdp=ans["sdp"], type="answer")
        )

    async def _on_remote_ice(self, d):
        try:
            wrap = json.loads(d["candidate"])
            c = wrap["candidate"]
            sdp_line = c["candidate"]
            if sdp_line.startswith("candidate:"):
                sdp_line = sdp_line[len("candidate:"):]
            cand = candidate_from_sdp(sdp_line)
            cand.sdpMid = c.get("sdpMid")
            cand.sdpMLineIndex = c.get("sdpMLineIndex")
            await self.pc.addIceCandidate(cand)
        except Exception as e:
            self._log(f"ICE error: {e}", "warning")

    async def _on_peer_disconnect(self, d):
        self._log("peer disconnected", "warning")
        await self._close_peer()
        if self.ws and not self.ws.closed and self._should_stay:
            self._set_state("searching")
            await self._search()

    async def skip(self):
        if self.connection_id:
            self._log("skip", "warning")
            try:
                await self._emit_event({
                    "type": "peer-disconnect",
                    "connectionId": self.connection_id,
                })
            except Exception:
                pass
        else:
            try: await self._emit_event({"type": "stop-scan"})
            except Exception: pass
        await self._close_peer()
        if self.ws and not self.ws.closed:
            self._set_state("searching")
            await self._search()

    async def disconnect(self):
        self._log("full disconnect")
        self._should_stay = False
        if self.connection_id:
            try:
                await self._emit_event({
                    "type": "peer-disconnect",
                    "connectionId": self.connection_id,
                })
            except Exception:
                pass
        await self._close_peer()
        if self._ping_task:
            self._ping_task.cancel()
        if self.ws:
            try: await self.ws.close()
            except Exception: pass
        self.ws = None
        self._set_state("idle")

    async def _close_peer(self):
        if self.pc:
            try: await self.pc.close()
            except Exception: pass
            self.pc = None
        self.connection_id = None


# ============================================================
# BRIDGE (hosted version)
# ============================================================
class Bridge:
    def __init__(self):
        self.relay = MediaRelay()
        self.mic_muted = True
        self.mic_target = "both"
        self.vol_a = 1.0
        self.vol_b = 1.0

        # выходные микшеры для собеседников и для браузера
        self.out_a = MixerTrack("out_A")
        self.out_b = MixerTrack("out_B")
        self.out_browser = MixerTrack("out_browser")  # → шлём в браузер пользователя

        # источник микрофона из браузера
        self.browser_mic = BrowserMicSource()

        # push-каналы для микрофона в каждый микшер
        mic_a = SourceChannel(None, "out_A/mic", preroll_ms=0)
        mic_b = SourceChannel(None, "out_B/mic", preroll_ms=0)
        self.browser_mic.subscribe(mic_a)
        self.browser_mic.subscribe(mic_b)
        self.out_a.add_source_channel("mic", mic_a, self._gain_mic_a)
        self.out_b.add_source_channel("mic", mic_b, self._gain_mic_b)

        self.session = None
        self.client_a = None
        self.client_b = None
        self._vu_task = None
        self.ws_clients: set = set()
        # WebSocket-клиенты, которые слушают audio-out (получают смешанный A+B)
        self.audio_out_clients: set = set()
        # задача которая периодически пулит из out_browser и рассылает клиентам
        self._audio_out_task = None

    async def start(self):
        log.info("Bridge.start: creating ClientSession...")
        self.session = aiohttp.ClientSession()

        self.client_a = NektoClient("A", self.out_a, self.relay,
                                     lambda tr: self._on_remote("a", tr),
                                     self._broadcast)
        self.client_b = NektoClient("B", self.out_b, self.relay,
                                     lambda tr: self._on_remote("b", tr),
                                     self._broadcast)

        self._vu_task = asyncio.create_task(self._vu_loop())
        self._audio_out_task = asyncio.create_task(self._audio_out_loop())
        log.info("Bridge.start: done")

    async def _audio_out_loop(self):
        """
        Читает по 20мс из out_browser, ресемплит 48k→16k, шлёт в WS.
        Оптимизация: параллельная отправка всем клиентам (один медленный не тормозит остальных).
        """
        down = av.AudioResampler(format="s16", layout="mono", rate=16000)
        try:
            while True:
                frame = await self.out_browser.recv()
                try:
                    out_frames = down.resample(frame)
                except Exception:
                    continue
                for f in out_frames:
                    arr = f.to_ndarray().flatten().astype(np.int16)
                    data = arr.tobytes()
                    if not self.audio_out_clients:
                        continue
                    # параллельная отправка — не ждём одного медленного
                    tasks = []
                    for ws in list(self.audio_out_clients):
                        tasks.append(self._send_audio_safe(ws, data))
                    if tasks:
                        await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.warning(f"audio_out_loop: {e}")

    async def _send_audio_safe(self, ws, data):
        try:
            await ws.send_bytes(data)
        except Exception:
            self.audio_out_clients.discard(ws)

    def _gain_mic_a(self):
        return 0 if self.mic_muted else (1 if self.mic_target in ("both", "a") else 0)

    def _gain_mic_b(self):
        return 0 if self.mic_muted else (1 if self.mic_target in ("both", "b") else 0)

    def _on_remote(self, side, track):
        opposite = self.out_b if side == "a" else self.out_a
        # На маршруте A↔B важнее низкая задержка чем стабильность: берём минимум preroll.
        opposite.add_source_channel(f"voice_{side}",
                                     SourceChannel(self.relay.subscribe(track),
                                                   f"opp/voice_{side}", preroll_ms=30),
                                     lambda: 1)
        # На маршруте A/B → твой браузер preroll чуть больше — трафик через интернет.
        vol_fn = (lambda: self.vol_a) if side == "a" else (lambda: self.vol_b)
        self.out_browser.add_source_channel(f"voice_{side}",
                                             SourceChannel(self.relay.subscribe(track),
                                                           f"brw/voice_{side}", preroll_ms=80),
                                             vol_fn)
        self._broadcast("log", {"side": "main", "level": "info",
                                 "msg": f"routed voice_{side}"})

    async def _vu_loop(self):
        while True:
            try:
                await asyncio.sleep(0.05)
                def norm(rms): return min(1.0, rms / 8000.0)
                mic_rms = self.browser_mic.current_rms
                voice_a = self.out_browser.source_rms.get("voice_a", 0.0)
                voice_b = self.out_browser.source_rms.get("voice_b", 0.0)
                self._broadcast("levels", {
                    "mic": norm(mic_rms), "a": norm(voice_a), "b": norm(voice_b),
                })
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.debug(f"vu loop: {e}")

    # ---------- commands from UI ----------
    async def cmd_connect(self, side, user_id, user_agent, criteria):
        c = self.client_a if side == "a" else self.client_b
        await c.connect(user_id, user_agent, criteria, self.session)

    async def cmd_skip(self, side):
        c = self.client_a if side == "a" else self.client_b
        await c.skip()

    async def cmd_disconnect(self, side):
        c = self.client_a if side == "a" else self.client_b
        await c.disconnect()

    def cmd_mute(self, val):
        self.mic_muted = bool(val)
        self._broadcast("mic", {"muted": self.mic_muted, "target": self.mic_target})

    def cmd_mic_target(self, val):
        self.mic_target = val
        self._broadcast("mic", {"muted": self.mic_muted, "target": self.mic_target})

    def cmd_volume(self, side, val):
        v = float(val)
        if side == "a": self.vol_a = v
        else: self.vol_b = v

    def _broadcast(self, evt, data=None):
        if not self.ws_clients: return
        msg = json.dumps({"evt": evt, "data": data})
        for ws in list(self.ws_clients):
            asyncio.create_task(self._send_safe(ws, msg))

    async def _send_safe(self, ws, msg):
        try: await ws.send_str(msg)
        except Exception: self.ws_clients.discard(ws)

    async def shutdown(self):
        if self._vu_task: self._vu_task.cancel()
        if self._audio_out_task: self._audio_out_task.cancel()
        if self.client_a: await self.client_a.disconnect()
        if self.client_b: await self.client_b.disconnect()
        if self.session: await self.session.close()


# ============================================================
# WEB UI
# ============================================================
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="ru"><head>
<meta charset="UTF-8"><title>Nekto Audio Bridge</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
:root{
  --bg:#1e1f22;
  --panel:#2b2d31;
  --panel2:#313338;
  --panel3:#383a40;
  --border:#1f2023;
  --divider:#3f4147;
  --text:#f2f3f5;
  --text-muted:#b5bac1;
  --dim:#80848e;
  --accent:#5865f2;
  --accent-h:#4752c4;
  --ok:#23a559;
  --ok-bright:#2ecc71;
  --warn:#f0b232;
  --err:#f23f42;
  --err-h:#da373c;
  --radius:8px;
  --radius-sm:4px;
  --shadow:0 2px 10px rgba(0,0,0,.3);
}
*{box-sizing:border-box}
body{margin:0;padding:16px;background:var(--bg);color:var(--text);
font:14px/1.4 "gg sans",-apple-system,"Helvetica Neue",Helvetica,Arial,sans-serif;
-webkit-font-smoothing:antialiased;min-height:100vh}
h1{margin:0;font-size:18px;font-weight:700;letter-spacing:-.02em}
h3{margin:0 0 12px;font-size:12px;color:var(--dim);
text-transform:uppercase;letter-spacing:.5px;font-weight:700;
display:flex;align-items:center;gap:8px}
.grid{display:grid;grid-template-columns:1fr 280px 1fr;gap:14px;max-width:1400px;margin:0 auto}
@media(max-width:900px){.grid{grid-template-columns:1fr}}
.panel{background:var(--panel);border-radius:var(--radius);
padding:16px;transition:box-shadow .1s,border-color .1s;
border:2px solid transparent;box-shadow:var(--shadow)}
.panel.speaking{border-color:var(--ok-bright);
box-shadow:0 0 0 2px var(--ok-bright),0 0 24px rgba(46,204,113,.35)}
.row{display:flex;gap:10px;align-items:center;margin:10px 0}
.row>label{flex:0 0 90px;color:var(--text-muted);font-size:12px;font-weight:500}
.row>*:not(label){flex:1}
button{background:var(--panel3);color:var(--text);border:none;
border-radius:var(--radius-sm);padding:10px 16px;cursor:pointer;
font-size:14px;font-weight:500;font-family:inherit;
transition:background .1s,opacity .1s;
display:inline-flex;align-items:center;justify-content:center;gap:6px;
min-height:38px}
button:hover{background:var(--divider)}
button:active{transform:scale(.98)}
button:disabled{opacity:.4;cursor:not-allowed;pointer-events:none}
button.primary{background:var(--accent);color:#fff}
button.primary:hover{background:var(--accent-h)}
button.danger{background:var(--err);color:#fff}
button.danger:hover{background:var(--err-h)}
button.ok{background:var(--ok);color:#fff}
button.ok:hover{background:#1a8c4a}
button.ghost{background:transparent;color:var(--text-muted)}
button.ghost:hover{background:var(--panel3);color:var(--text)}
button.iconbtn{width:42px;min-width:42px;padding:0}
button svg{flex-shrink:0}
select,input[type=text],textarea{background:var(--panel2);color:var(--text);
border:1px solid var(--border);border-radius:var(--radius-sm);
padding:10px 12px;font-size:13px;width:100%;font-family:inherit;
transition:border-color .1s}
select:focus,input[type=text]:focus{outline:none;border-color:var(--accent)}
input[type=range]{width:100%;height:8px;-webkit-appearance:none;appearance:none;
background:var(--panel3);border-radius:4px;outline:none}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;width:18px;height:18px;
background:#fff;border-radius:50%;cursor:pointer;box-shadow:0 2px 4px rgba(0,0,0,.4)}
input[type=range]::-moz-range-thumb{width:18px;height:18px;background:#fff;
border-radius:50%;cursor:pointer;border:0;box-shadow:0 2px 4px rgba(0,0,0,.4)}
.status{padding:12px 14px;border-radius:var(--radius-sm);margin-bottom:12px;
min-height:42px;background:var(--panel2);font-size:13px;color:var(--text-muted);
display:flex;align-items:center;gap:8px;font-weight:500}
.status.s{color:var(--warn);background:rgba(240,178,50,.08)}
.status.t{color:var(--ok-bright);background:rgba(46,204,113,.08)}
.btn-row{display:flex;gap:6px;margin-top:12px}
.btn-row>button:first-child{flex:1}
.timer{font-size:30px;text-align:center;font-family:"gg mono",monospace;
margin:12px 0;color:var(--text);font-weight:600;letter-spacing:.05em}
.mtarget{display:grid;grid-template-columns:1fr 1fr;gap:4px;margin:6px 0}
.mtarget input{display:none}
.mtarget label{padding:9px;text-align:center;background:var(--panel2);
border-radius:var(--radius-sm);cursor:pointer;font-size:12px;font-weight:500;
color:var(--text-muted);transition:all .1s}
.mtarget label:hover{background:var(--panel3);color:var(--text)}
.mtarget input:checked+label{background:var(--accent);color:#fff}
.log{background:#1a1b1e;color:#a0e6a0;font-family:"gg mono",monospace;
font-size:11px;padding:10px;border-radius:var(--radius-sm);height:160px;
overflow-y:auto;white-space:pre-wrap;word-break:break-all;line-height:1.4;
border:1px solid var(--border)}
.log .err{color:#ff6b6b}.log .warn{color:#ffd93d}.log .ok{color:#6eff7f}
details{margin-top:12px}
summary{cursor:pointer;color:var(--dim);font-size:12px;user-select:none;
padding:6px 0;font-weight:500}
summary:hover{color:var(--text-muted)}
details[open] summary{color:var(--text-muted);margin-bottom:8px}
.speak-dot{display:inline-block;width:12px;height:12px;border-radius:50%;
background:var(--dim);transition:background .08s}
.speak-dot.on{background:var(--ok-bright);animation:pulse 1.5s infinite}
@keyframes pulse{
  0%{box-shadow:0 0 0 0 rgba(46,204,113,.7)}
  70%{box-shadow:0 0 0 10px rgba(46,204,113,0)}
  100%{box-shadow:0 0 0 0 rgba(46,204,113,0)}
}
.warning-box{background:rgba(240,178,50,.08);border:1px solid rgba(240,178,50,.3);
color:var(--warn);padding:12px 14px;border-radius:var(--radius-sm);
margin-bottom:14px;font-size:12px;line-height:1.5}
kbd{background:var(--panel2);border:1px solid var(--divider);
border-radius:3px;padding:2px 6px;font-family:monospace;font-size:11px;color:var(--text)}
#startAudioBtn{width:100%;margin-bottom:14px;padding:14px;font-size:15px;font-weight:600}
#audioDot{width:10px;height:10px;border-radius:50%;background:var(--err);
display:inline-block;margin-right:6px;vertical-align:middle}
#audioDot.ok{background:var(--ok-bright)}
.app-header{padding:14px 18px;background:var(--panel);border-radius:var(--radius);
margin-bottom:16px;display:flex;align-items:center;gap:14px;box-shadow:var(--shadow);max-width:1400px;margin-left:auto;margin-right:auto}
.app-logo{width:40px;height:40px;border-radius:50%;background:var(--accent);
display:flex;align-items:center;justify-content:center;flex-shrink:0}
.app-title{flex:1}
.app-status{font-size:12px;color:var(--text-muted);margin-top:3px;display:flex;align-items:center}
#muteBtn{width:100%;padding:14px;font-size:14px;font-weight:600;
display:flex;gap:10px;align-items:center;justify-content:center}
.icon{width:18px;height:18px;flex-shrink:0}
.icon-sm{width:14px;height:14px;flex-shrink:0}
.icon-lg{width:22px;height:22px;flex-shrink:0}
</style></head>
<body>

<div class="app-header">
  <div class="app-logo">
    <svg class="icon-lg" viewBox="0 0 24 24" fill="#fff"><path d="M12 14a3 3 0 003-3V5a3 3 0 00-6 0v6a3 3 0 003 3z"/><path d="M19 11a1 1 0 00-2 0 5 5 0 01-10 0 1 1 0 00-2 0 7 7 0 006 6.92V21a1 1 0 002 0v-3.08A7 7 0 0019 11z"/></svg>
  </div>
  <div class="app-title">
    <h1>Nekto Audio Bridge</h1>
    <div class="app-status"><span id="audioDot"></span><span id="audioStatus">аудио не запущено</span></div>
  </div>
</div>

<div style="max-width:1400px;margin:0 auto">
<div class="warning-box">
<b>Инструкция:</b> 1) пройди 3 капчи в <kbd>nekto.me/audiochat</kbd> в обоих аккаунтах,
2) в консоли F12 выполни <code style="font-size:11px">JSON.parse(localStorage.getItem("storage_audio_v2")).user.authToken</code>,
3) скопируй UUID и <code>navigator.userAgent</code> в поля ниже,
4) нажми <b>"Включить микрофон"</b>.
</div>

<button id="startAudioBtn" class="primary">
  <svg class="icon" viewBox="0 0 24 24" fill="currentColor"><path d="M12 14a3 3 0 003-3V5a3 3 0 00-6 0v6a3 3 0 003 3z"/><path d="M19 11a1 1 0 00-2 0 5 5 0 01-10 0 1 1 0 00-2 0 7 7 0 006 6.92V21a1 1 0 002 0v-3.08A7 7 0 0019 11z"/></svg>
  Включить микрофон и аудио
</button>
</div>

<div class="grid">

<div class="panel" id="panelA">
<h3>
  <svg class="icon-sm" viewBox="0 0 24 24" fill="currentColor"><path d="M12 12a5 5 0 100-10 5 5 0 000 10zM21 21a9 9 0 00-18 0h18z"/></svg>
  Собеседник A <span id="speakDotA" class="speak-dot"></span>
</h3>
<div id="statusA" class="status">Не подключено</div>

<div class="row"><label>authToken:</label>
<input type="text" id="tokenA" placeholder="UUID" style="font-family:monospace;font-size:11px"></div>
<div class="row"><label>UA:</label>
<input type="text" id="uaA" placeholder="navigator.userAgent" style="font-size:11px"></div>

<details open><summary>🎯 Параметры поиска</summary>
<div class="row"><label>Мой пол:</label>
<select id="userSexA"><option value="MALE">Мужской</option><option value="FEMALE">Женский</option></select></div>
<div class="row"><label>Мой возр:</label>
<select id="userAgeA"><option value="0-17">до 17</option><option value="18-24">18-24</option><option value="25-32" selected>25-32</option><option value="33-100">33+</option></select></div>
<div class="row"><label>Пол парт:</label>
<select id="peerSexA"><option value="ANY">Не важно</option><option value="MALE">Мужской</option><option value="FEMALE">Женский</option></select></div>
<div class="row"><label>Возр парт:</label>
<input type="text" id="peerAgeA" value="18-24,25-32,33-100"></div>
</details>

<div class="btn-row">
<button id="connA" class="primary">
  <svg class="icon-sm" viewBox="0 0 24 24" fill="currentColor"><path d="M20 18.35l-8-6.2-8 6.2V5a1 1 0 011-1h14a1 1 0 011 1v13.35z" transform="rotate(180 12 12)"/></svg>
  Подключить
</button>
<button id="skipA" class="ghost iconbtn" disabled title="Скип">
  <svg class="icon" viewBox="0 0 24 24" fill="currentColor"><path d="M4 5v14l8-7L4 5zm10 0v14l8-7-8-7z"/></svg>
</button>
<button id="discA" class="danger iconbtn" disabled title="Отключить">
  <svg class="icon" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2a1 1 0 011 1v9a1 1 0 01-2 0V3a1 1 0 011-1zm-4.95 3.05a1 1 0 010 1.42 7 7 0 109.9 0 1 1 0 111.41-1.42 9 9 0 11-12.73 0 1 1 0 011.42 0z"/></svg>
</button>
</div>

<div class="row" style="margin-top:14px"><label>Громкость:</label>
<input type="range" id="volA" min="0" max="200" value="100">
<span id="volLabelA" style="flex:0 0 40px;text-align:right;font-size:11px;color:var(--text-muted)">100%</span>
</div>
<details><summary>Лог A</summary><div id="logA" class="log"></div></details>
</div>

<div class="panel" id="panelMic">
<h3>
  <svg class="icon-sm" viewBox="0 0 24 24" fill="currentColor"><path d="M12 3c-1.1 0-2 .9-2 2v6c0 1.1.9 2 2 2s2-.9 2-2V5c0-1.1-.9-2-2-2zm5 8c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/></svg>
  Пульт <span id="speakDotMic" class="speak-dot"></span>
</h3>
<div class="timer" id="timer">00:00</div>
<button id="muteBtn" class="danger">
  <svg class="icon" viewBox="0 0 24 24" fill="currentColor"><path d="M19 11h-1.7c0 .74-.16 1.43-.43 2.05l1.23 1.23c.56-.98.9-2.09.9-3.28zM14.98 11.17c0-.06.02-.11.02-.17V5c0-1.66-1.34-3-3-3S9 3.34 9 5v.18l5.98 5.99zM4.27 3L3 4.27l6.01 6.01V11c0 1.66 1.33 3 2.99 3 .22 0 .44-.03.65-.08l1.66 1.66c-.71.33-1.5.52-2.31.52-2.76 0-5.3-2.1-5.3-5.1H5c0 3.41 2.72 6.23 6 6.72V21h2v-3.28c.91-.13 1.77-.45 2.54-.9L19.73 21 21 19.73 4.27 3z"/></svg>
  Микрофон замьючен
</button>

<div style="margin-top:14px">
<div style="color:var(--text-muted);font-size:12px;margin-bottom:8px;font-weight:500">Меня слышат:</div>
<div class="mtarget">
<input type="radio" name="mtgt" id="mt1" value="both" checked><label for="mt1">Оба</label>
<input type="radio" name="mtgt" id="mt2" value="none"><label for="mt2">Никто</label>
<input type="radio" name="mtgt" id="mt3" value="a"><label for="mt3">Только A</label>
<input type="radio" name="mtgt" id="mt4" value="b"><label for="mt4">Только B</label>
</div></div>
<details open><summary>Общий лог</summary><div id="logMain" class="log" style="height:200px;margin-top:6px"></div></details>
</div>

<div class="panel" id="panelB">
<h3>
  <svg class="icon-sm" viewBox="0 0 24 24" fill="currentColor"><path d="M12 12a5 5 0 100-10 5 5 0 000 10zM21 21a9 9 0 00-18 0h18z"/></svg>
  Собеседник B <span id="speakDotB" class="speak-dot"></span>
</h3>
<div id="statusB" class="status">Не подключено</div>

<div class="row"><label>authToken:</label>
<input type="text" id="tokenB" placeholder="UUID" style="font-family:monospace;font-size:11px"></div>
<div class="row"><label>UA:</label>
<input type="text" id="uaB" placeholder="navigator.userAgent" style="font-size:11px"></div>

<details open><summary>🎯 Параметры поиска</summary>
<div class="row"><label>Мой пол:</label>
<select id="userSexB"><option value="MALE">Мужской</option><option value="FEMALE">Женский</option></select></div>
<div class="row"><label>Мой возр:</label>
<select id="userAgeB"><option value="0-17">до 17</option><option value="18-24">18-24</option><option value="25-32" selected>25-32</option><option value="33-100">33+</option></select></div>
<div class="row"><label>Пол парт:</label>
<select id="peerSexB"><option value="ANY">Не важно</option><option value="MALE">Мужской</option><option value="FEMALE">Женский</option></select></div>
<div class="row"><label>Возр парт:</label>
<input type="text" id="peerAgeB" value="18-24,25-32,33-100"></div>
</details>

<div class="btn-row">
<button id="connB" class="primary">
  <svg class="icon-sm" viewBox="0 0 24 24" fill="currentColor"><path d="M20 18.35l-8-6.2-8 6.2V5a1 1 0 011-1h14a1 1 0 011 1v13.35z" transform="rotate(180 12 12)"/></svg>
  Подключить
</button>
<button id="skipB" class="ghost iconbtn" disabled title="Скип">
  <svg class="icon" viewBox="0 0 24 24" fill="currentColor"><path d="M4 5v14l8-7L4 5zm10 0v14l8-7-8-7z"/></svg>
</button>
<button id="discB" class="danger iconbtn" disabled title="Отключить">
  <svg class="icon" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2a1 1 0 011 1v9a1 1 0 01-2 0V3a1 1 0 011-1zm-4.95 3.05a1 1 0 010 1.42 7 7 0 109.9 0 1 1 0 111.41-1.42 9 9 0 11-12.73 0 1 1 0 011.42 0z"/></svg>
</button>
</div>

<div class="row" style="margin-top:14px"><label>Громкость:</label>
<input type="range" id="volB" min="0" max="200" value="100">
<span id="volLabelB" style="flex:0 0 40px;text-align:right;font-size:11px;color:var(--text-muted)">100%</span>
</div>
<details><summary>Лог B</summary><div id="logB" class="log"></div></details>
</div>

</div>

<script>
let ws=null;
let timerStart=null,timerIv=null;
const states={a:'idle',b:'idle'};

// localStorage
const save=(k,v)=>localStorage.setItem('nekto_'+k,v);
const load=(k,d='')=>localStorage.getItem('nekto_'+k)||d;
['A','B'].forEach(s=>{
  ['token','ua','peerAge','userSex','userAge','peerSex'].forEach(f=>{
    const el=document.getElementById(f+s);
    if(!el)return;
    const k=f+s;
    el.value=load(k,el.value);
    el.addEventListener('change',()=>save(k,el.value));
    el.addEventListener('input',()=>save(k,el.value));
  });
});

function logLine(target,msg,cls=''){
  const el=document.getElementById(target);if(!el)return;
  const t=new Date().toTimeString().slice(0,8);
  const line=document.createElement('div');
  if(cls)line.className=cls;
  line.textContent=`[${t}] ${msg}`;
  el.appendChild(line);el.scrollTop=el.scrollHeight;
  while(el.children.length>300)el.removeChild(el.firstChild);
}

function connectWS(){
  const proto=location.protocol==='https:'?'wss:':'ws:';
  ws=new WebSocket(`${proto}//${location.host}/ws`);
  ws.onopen=()=>logLine('logMain','Connected to server','ok');
  ws.onclose=()=>{
    logLine('logMain','Disconnected from server','err');
    setTimeout(connectWS,2000);
  };
  ws.onmessage=e=>{
    const {evt,data}=JSON.parse(e.data);
    handleEvent(evt,data);
  };
}

function handleEvent(evt,data){
  if(evt==='log'){
    const target=data.side==='main'?'logMain':
                 data.side==='a'?'logA':
                 data.side==='b'?'logB':'logMain';
    logLine(target,data.msg,data.level);
    if(target!=='logMain')logLine('logMain',`[${data.side.toUpperCase()}] ${data.msg}`,data.level);
  }else if(evt==='state'){
    states[data.side]=data.state;
    updateStatus(data.side,data.state);
    updateTimer();
  }else if(evt==='mic'){
    updateMicBtn(data.muted);
  }else if(evt==='levels'){
    updateSpeaking('a',data.a);
    updateSpeaking('b',data.b);
    updateSpeaking('mic',data.mic);
  }
}

const SPEAK_ON=0.025,SPEAK_OFF=0.012,TAIL=180;
const speakState={a:false,b:false,mic:false};
const speakLast={a:0,b:0,mic:0};

function updateSpeaking(which,level){
  const now=Date.now();
  const was=speakState[which];
  let is;
  if(level>SPEAK_ON){is=true;speakLast[which]=now;}
  else if(level<SPEAK_OFF){is=(now-speakLast[which])<TAIL;}
  else{is=was;}
  if(is===was)return;
  speakState[which]=is;
  const dot=document.getElementById(which==='mic'?'speakDotMic':which==='a'?'speakDotA':'speakDotB');
  const panel=document.getElementById(which==='mic'?'panelMic':which==='a'?'panelA':'panelB');
  if(is){dot.classList.add('on');panel.classList.add('speaking');}
  else{dot.classList.remove('on');panel.classList.remove('speaking');}
}

function updateStatus(side,state){
  const el=document.getElementById('status'+side.toUpperCase());
  const connBtn=document.getElementById('conn'+side.toUpperCase());
  const skipBtn=document.getElementById('skip'+side.toUpperCase());
  const discBtn=document.getElementById('disc'+side.toUpperCase());
  el.classList.remove('s','t');
  const map={idle:'Не подключено',connecting:'⚡ Подключаюсь...',
    searching:'🔍 Поиск...','peer-found':'⚡ Соединение...',talking:'🎤 В разговоре'};
  el.textContent=map[state]||state;
  if(['searching','peer-found','connecting'].includes(state))el.classList.add('s');
  if(state==='talking')el.classList.add('t');
  const busy=state!=='idle';
  connBtn.disabled=busy;skipBtn.disabled=!busy;discBtn.disabled=!busy;
}

function updateTimer(){
  const talk=states.a==='talking'||states.b==='talking';
  if(talk&&!timerStart){
    timerStart=Date.now();
    timerIv=setInterval(()=>{
      const s=Math.floor((Date.now()-timerStart)/1000);
      document.getElementById('timer').textContent=
        String(Math.floor(s/60)).padStart(2,'0')+':'+String(s%60).padStart(2,'0');
    },250);
  }else if(!talk&&timerStart){
    clearInterval(timerIv);timerIv=null;timerStart=null;
    document.getElementById('timer').textContent='00:00';
  }
}

function updateMicBtn(muted){
  const btn=document.getElementById('muteBtn');
  if(muted){btn.textContent='🔇 Микрофон замьючен';btn.className='danger';}
  else{btn.textContent='🎙 Микрофон активен';btn.className='ok';}
}

function send(cmd){
  if(ws&&ws.readyState===1)ws.send(JSON.stringify(cmd));
  else logLine('logMain','WS disconnected','err');
}

function parseAges(s){
  return s.split(',').map(x=>x.trim()).filter(Boolean).map(x=>{
    const [f,t]=x.split('-').map(Number);return {from:f,to:t};
  });
}
function readCriteria(side){
  const S=side.toUpperCase();
  const [uf,ut]=document.getElementById('userAge'+S).value.split('-').map(Number);
  return {
    group:0,
    userSex:document.getElementById('userSex'+S).value,
    peerSex:document.getElementById('peerSex'+S).value,
    userAge:{from:uf,to:ut},
    peerAges:parseAges(document.getElementById('peerAge'+S).value),
  };
}

for(const side of ['A','B']){
  document.getElementById('conn'+side).onclick=()=>{
    const token=document.getElementById('token'+side).value.trim();
    const ua=document.getElementById('ua'+side).value.trim();
    if(!token){logLine('logMain',`[${side}] вставь authToken`,'err');return;}
    if(!ua){logLine('logMain',`[${side}] вставь UA`,'err');return;}
    send({cmd:'connect',side:side.toLowerCase(),user_id:token,user_agent:ua,criteria:readCriteria(side)});
  };
  document.getElementById('skip'+side).onclick=()=>send({cmd:'skip',side:side.toLowerCase()});
  document.getElementById('disc'+side).onclick=()=>send({cmd:'disconnect',side:side.toLowerCase()});
  const vol=document.getElementById('vol'+side);
  vol.oninput=()=>{
    document.getElementById('volLabel'+side).textContent=vol.value+'%';
    send({cmd:'volume',side:side.toLowerCase(),value:vol.value/100});
  };
}

let micMuted=true;
document.getElementById('muteBtn').onclick=()=>{
  micMuted=!micMuted;
  send({cmd:'mute',value:micMuted});
  updateMicBtn(micMuted);
};
document.querySelectorAll('input[name=mtgt]').forEach(r=>{
  r.onchange=()=>send({cmd:'mic_target',value:document.querySelector('input[name=mtgt]:checked').value});
});

// ===== WebSocket-аудио к серверу =====
let audioCtx=null;
let micNode=null;
let playerNode=null;
let micWS=null;
let outWS=null;

async function startAudio(){
  const btn=document.getElementById('startAudioBtn');
  btn.disabled=true;btn.textContent='⏳ Запуск...';
  try{
    // 1. Микрофон
    const stream=await navigator.mediaDevices.getUserMedia({
      audio:{echoCancellation:true,noiseSuppression:true,autoGainControl:true,sampleRate:16000}
    });

    // 2. AudioContext 16kHz
    audioCtx=new (window.AudioContext||window.webkitAudioContext)({sampleRate:16000,latencyHint:'playback'});
    if(audioCtx.state==='suspended')await audioCtx.resume();

    // 3. Два AudioWorklet — один для захвата микрофона, другой для воспроизведения
    const workletCode=`
class MicProcessor extends AudioWorkletProcessor{
  constructor(){
    super();
    this.buffer=new Int16Array(320); // 20ms @ 16kHz
    this.pos=0;
  }
  process(inputs){
    const input=inputs[0];
    if(!input||!input[0])return true;
    const ch=input[0];
    for(let i=0;i<ch.length;i++){
      const s=Math.max(-1,Math.min(1,ch[i]));
      this.buffer[this.pos++]=s<0?s*0x8000:s*0x7fff;
      if(this.pos>=this.buffer.length){
        this.port.postMessage(this.buffer.slice());
        this.pos=0;
      }
    }
    return true;
  }
}
registerProcessor('mic-processor',MicProcessor);

// Плеер — кольцевой буфер сэмплов, читает ровно столько сколько нужно ОС.
// Сглаживает джиттер сети, не создаёт "клики" между кусочками.
class PlayerProcessor extends AudioWorkletProcessor{
  constructor(){
    super();
    this.buf=new Float32Array(16000*4); // 4 секунды буфера
    this.writePos=0;
    this.readPos=0;
    this.available=0;
    // preroll: ждём пока накопится 400мс прежде чем начнём выдавать
    this.preroll=16000*0.4|0;
    this.prerolled=false;
    // статистика для подстройки
    this.tooFull=16000*1.2|0;   // если > 1.2s — ускоримся (дропнем)
    this.tooEmpty=16000*0.1|0;   // если < 100ms после preroll — снова prerolled=false
    this.port.onmessage=(e)=>{
      const arr=e.data;
      // пишем в кольцевой буфер
      const n=arr.length;
      const bl=this.buf.length;
      if(this.available+n>bl){
        // дропаем старейшее
        const drop=this.available+n-bl;
        this.readPos=(this.readPos+drop)%bl;
        this.available-=drop;
      }
      const end=this.writePos+n;
      if(end<=bl){
        this.buf.set(arr,this.writePos);
      }else{
        const first=bl-this.writePos;
        this.buf.set(arr.subarray(0,first),this.writePos);
        this.buf.set(arr.subarray(first),0);
      }
      this.writePos=(this.writePos+n)%bl;
      this.available+=n;
      // если переполняемся — ускоренно дренируем
      if(this.available>this.tooFull){
        const drop=this.available-this.preroll;
        this.readPos=(this.readPos+drop)%bl;
        this.available-=drop;
      }
    };
  }
  process(_,outputs){
    const out=outputs[0][0];
    const n=out.length;
    const bl=this.buf.length;
    if(!this.prerolled){
      if(this.available>=this.preroll)this.prerolled=true;
      else{out.fill(0);return true;}
    }
    if(this.available<n){
      // underrun — тишина, сбрасываем preroll
      out.fill(0);
      this.prerolled=false;
      return true;
    }
    const end=this.readPos+n;
    if(end<=bl){
      out.set(this.buf.subarray(this.readPos,end));
    }else{
      const first=bl-this.readPos;
      out.set(this.buf.subarray(this.readPos),0);
      out.set(this.buf.subarray(0,n-first),first);
    }
    this.readPos=(this.readPos+n)%bl;
    this.available-=n;
    return true;
  }
}
registerProcessor('player-processor',PlayerProcessor);
`;
    const blob=new Blob([workletCode],{type:'application/javascript'});
    const url=URL.createObjectURL(blob);
    await audioCtx.audioWorklet.addModule(url);

    // ---- микрофон ----
    const source=audioCtx.createMediaStreamSource(stream);
    micNode=new AudioWorkletNode(audioCtx,'mic-processor');
    source.connect(micNode);
    const silent=audioCtx.createGain();
    silent.gain.value=0;
    micNode.connect(silent);
    silent.connect(audioCtx.destination);

    // ---- плеер ----
    playerNode=new AudioWorkletNode(audioCtx,'player-processor');
    playerNode.connect(audioCtx.destination);

    // 4. WebSocket для микрофона
    const proto=location.protocol==='https:'?'wss:':'ws:';
    micWS=new WebSocket(`${proto}//${location.host}/audio-in`);
    micWS.binaryType='arraybuffer';

    micNode.port.onmessage=(e)=>{
      if(micWS&&micWS.readyState===1){
        micWS.send(e.data.buffer);
      }
    };

    micWS.onopen=()=>logLine('logMain','mic WS connected','ok');
    micWS.onerror=()=>logLine('logMain','mic WS error','err');
    micWS.onclose=()=>logLine('logMain','mic WS closed','warn');

    // 5. WebSocket для получения звука — шлём прямо в плеер-воркплет
    outWS=new WebSocket(`${proto}//${location.host}/audio-out`);
    outWS.binaryType='arraybuffer';

    outWS.onopen=()=>logLine('logMain','audio-out WS connected','ok');
    outWS.onmessage=(e)=>{
      // конвертируем int16 → float32 и шлём плееру
      const int16=new Int16Array(e.data);
      const float32=new Float32Array(int16.length);
      for(let i=0;i<int16.length;i++)float32[i]=int16[i]/32768;
      playerNode.port.postMessage(float32,[float32.buffer]);
    };
    outWS.onerror=()=>logLine('logMain','audio-out WS error','err');
    outWS.onclose=()=>logLine('logMain','audio-out WS closed','warn');

    document.getElementById('audioDot').classList.add('ok');
    document.getElementById('audioStatus').textContent='аудио работает';
    btn.textContent='✓ Аудио подключено';
    btn.className='ok';
  }catch(e){
    logLine('logMain','startAudio error: '+e.message,'err');
    btn.disabled=false;
    btn.textContent='🎤 Включить микрофон и аудио';
  }
}
document.getElementById('startAudioBtn').onclick=startAudio;

connectWS();
</script>
</body></html>
"""# ============================================================
# HTTP HANDLERS
# ============================================================
async def index_handler(request):
    return web.Response(text=INDEX_HTML, content_type="text/html")


async def audio_in_handler(request):
    """Браузер шлёт свой микрофон сюда (PCM int16 mono @ 16kHz)."""
    ws = web.WebSocketResponse(max_msg_size=0)  # без лимита
    await ws.prepare(request)
    bridge: Bridge = request.app["bridge"]
    log.info("audio-in client connected")
    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.BINARY:
                bridge.browser_mic.push_pcm(msg.data, input_rate=16000)
            elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR,
                              aiohttp.WSMsgType.CLOSED):
                break
    except Exception as e:
        log.debug(f"audio-in: {e}")
    finally:
        log.info("audio-in client disconnected")
    return ws


async def audio_out_handler(request):
    """Отдаём смешанный A+B звук браузеру (PCM int16 mono @ 16kHz)."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    bridge: Bridge = request.app["bridge"]
    bridge.audio_out_clients.add(ws)
    log.info("audio-out client connected")
    try:
        # удерживаем соединение открытым; Bridge сам шлёт в ws через _audio_out_loop
        async for msg in ws:
            if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR,
                            aiohttp.WSMsgType.CLOSED):
                break
    except Exception as e:
        log.debug(f"audio-out: {e}")
    finally:
        bridge.audio_out_clients.discard(ws)
        log.info("audio-out client disconnected")
    return ws


async def ws_handler(request):
    ws = web.WebSocketResponse(heartbeat=30)
    await ws.prepare(request)
    bridge: Bridge = request.app["bridge"]
    bridge.ws_clients.add(ws)
    try:
        await ws.send_str(json.dumps({"evt": "init", "data": {"status": "ok"}}))
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    c = json.loads(msg.data)
                    await dispatch(bridge, c)
                except Exception as e:
                    log.warning(f"ws cmd error: {e}")
    except Exception as e:
        log.debug(f"ws handler: {e}")
    finally:
        bridge.ws_clients.discard(ws)
    return ws


async def dispatch(bridge: Bridge, c):
    cmd = c.get("cmd")
    if cmd == "connect":
        await bridge.cmd_connect(c["side"], c["user_id"], c["user_agent"], c["criteria"])
    elif cmd == "skip":
        await bridge.cmd_skip(c["side"])
    elif cmd == "disconnect":
        await bridge.cmd_disconnect(c["side"])
    elif cmd == "mute":
        bridge.cmd_mute(c["value"])
    elif cmd == "mic_target":
        bridge.cmd_mic_target(c["value"])
    elif cmd == "volume":
        bridge.cmd_volume(c["side"], c["value"])


# ============================================================
# MAIN
# ============================================================
async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    )

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8765"))

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--port":
            port = int(args[i+1]); i += 2
        elif args[i] == "--host":
            host = args[i+1]; i += 2
        else:
            i += 1

    print(f">>> Nekto Bridge (hosted) starting on {host}:{port}")
    bridge = Bridge()
    await bridge.start()

    app = web.Application()
    app["bridge"] = bridge
    app.router.add_get("/", index_handler)
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/audio-in", audio_in_handler)
    app.router.add_get("/audio-out", audio_out_handler)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()

    print(f">>> Ready. Open http://{host}:{port} in browser")
    print(">>> Ctrl+C to stop")

    try:
        await asyncio.Future()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await bridge.shutdown()
        await runner.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
