# GPU backend image. Build from the repository root:
#   docker build -f backend/Dockerfile -t retail-vision-backend .
#
# CUDA 12.6 runtime + Python 3.12 (Ubuntu 24.04) matches
# torch==2.13.0 / torchvision==0.28.0 cu126 wheels.

FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.hf_home

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

COPY backend/requirements.txt /app/backend/requirements.txt
RUN python -m pip install --break-system-packages --no-cache-dir \
        torch==2.13.0 torchvision==0.28.0 \
        --index-url https://download.pytorch.org/whl/cu126 \
    && python -m pip install --break-system-packages --no-cache-dir \
        -r /app/backend/requirements.txt

COPY backend /app/backend
COPY products /app/products
COPY frontend /app/frontend
COPY pyproject.toml /app/pyproject.toml

ENV PYTHONPATH=/app
EXPOSE 8000

CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
