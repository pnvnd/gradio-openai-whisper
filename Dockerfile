# Build this docker image with:
# docker build -t transcription:latest .
# Use a Debian-based slim image
FROM python:3.13-slim-bookworm

WORKDIR /app

# Install build dependencies using apt-get
# gcc, g++, and libffi-dev are needed for compiling C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install packages
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python3 -m pip install --no-cache-dir openai-whisper gradio --extra-index-url https://download.pytorch.org/whl/cpu

COPY transcription.py /app/

# Download base model cache
RUN curl -JLO https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt

EXPOSE 7860

ENTRYPOINT ["python3", "transcription.py"]

# To run the container, use:
# docker run --name openai-whisper -dp 7860:7860 transcription:latest

# Once the container is running, go to http://localhost:7860