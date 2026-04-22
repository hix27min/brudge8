FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libopus0 libvpx7 libsrtp2-1 libavdevice59 \
    libavformat59 libavcodec59 libavutil57 libswscale6 libswresample4 \
    build-essential pkg-config \
    libopus-dev libvpx-dev libsrtp2-dev libavdevice-dev \
    libavfilter-dev libssl-dev ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY nekto_bridge_hosted.py .

ENV HOST=0.0.0.0
ENV PORT=8765
EXPOSE 8765

CMD ["python", "-u", "nekto_bridge_hosted.py"]
