FROM python:3.11-slim

# -------------------------------
# Python env
# -------------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.cache/huggingface

# -------------------------------
# System deps (opencv, ffmpeg, transformers)
# -------------------------------
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------
# App
# -------------------------------
WORKDIR /app

COPY requirements.txt .

# Replace opencv-python with opencv-python-headless for Docker
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y opencv-python \
    && pip install --no-cache-dir opencv-python-headless==4.12.0.88

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
