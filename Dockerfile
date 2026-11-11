# Kagu server. The Line Art engine is copied into ./engine before the build:
#   engine/lineart/   (the lineart folder of the scripts repository)
#   engine/flat2svg/  (the flat2svg folder of the scripts repository)
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends librsvg2-bin && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY server ./server
COPY web ./web
COPY engine /engine
ENV KAGU_ENGINE_DIR=/engine KAGU_DATA_DIR=/tmp/kagu PORT=8080
CMD exec uvicorn server.app:app --host 0.0.0.0 --port ${PORT}
