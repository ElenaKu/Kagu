# Kagu server. The Line Art engine is copied into ./engine before the build:
#   engine/lineart/   (the lineart folder of the scripts repository)
#   engine/flat2svg/  (the flat2svg folder of the scripts repository)
# Works on Hugging Face Spaces (port 7860, user 1000) and on any Docker host (set PORT).
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends librsvg2-bin && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=user server ./server
COPY --chown=user web ./web
COPY --chown=user engine /engine
USER user
ENV KAGU_ENGINE_DIR=/engine KAGU_DATA_DIR=/tmp/kagu PORT=7860
CMD exec uvicorn server.app:app --host 0.0.0.0 --port ${PORT}
