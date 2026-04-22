FROM python:3.11-slim

# Системные зависимости для aiortc
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

# HOST обязательно 0.0.0.0 (не localhost!), иначе внутри контейнера извне не достучаться
ENV HOST=0.0.0.0
# PORT НЕ задаём — Render подставит свой через переменную окружения

CMD ["python", "-u", "nekto_bridge_hosted.py"]
