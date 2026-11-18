# Kagu server. The Line Art engine is copied into ./engine before the build:
#   engine/lineart/   (a checkout of github.com/ElenaKu/lineart)
#   engine/flat2svg/  (a checkout of github.com/ElenaKu/flat2svg, branch core-v2)
# Small pictures are enlarged by an x4 ONNX model (Upscayl does not exist on the server).
# UPSCALER_URL can be changed at build time: docker build --build-arg UPSCALER_URL=...
# Works on Hugging Face Spaces (port 7860, user 1000) and on any Docker host (set PORT).
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends librsvg2-bin && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
ARG UPSCALER_URL=https://huggingface.co/qualcomm/Real-ESRGAN-x4plus/resolve/main/Real-ESRGAN-x4plus.onnx
ADD ${UPSCALER_URL} /models/upscaler-x4.onnx
RUN chmod 644 /models/upscaler-x4.onnx
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=user server ./server
COPY --chown=user web ./web
COPY --chown=user engine /engine
USER user
ENV KAGU_ENGINE_DIR=/engine KAGU_DATA_DIR=/tmp/kagu PORT=7860 \
    LINEART_UPSCAYL= LINEART_ONNX_UPSCALER=/models/upscaler-x4.onnx
CMD exec uvicorn server.app:app --host 0.0.0.0 --port ${PORT}
