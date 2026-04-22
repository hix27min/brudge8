FROM python:3.11-slim

# Системные зависимости для aiortc (WebRTC, Opus, VP8, SRTP, FFmpeg)
# Используем -dev пакеты — они потянут за собой нужные рантайм-версии автоматически,
# независимо от того Debian 11 или 12
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libopus-dev \
    libvpx-dev \
    libsrtp2-dev \
    libavdevice-dev \
    libavfilter-dev \
    libavformat-dev \
    libavcodec-dev \
    libswscale-dev \
    libswresample-dev \
    libavutil-dev \
    libssl-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY nekto_bridge_hosted.py .

ENV HOST=0.0.0.0
ENV PORT=8765
EXPOSE 8765

CMD ["python", "-u", "nekto_bridge_hosted.py"]
