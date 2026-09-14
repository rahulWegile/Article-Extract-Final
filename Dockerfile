# GPU-accelerated backend image.
#
# Base is the plain NVIDIA CUDA runtime, not `pytorch/pytorch:2.4.0-
# cuda12.4-cudnn9-runtime` -- that image pins its own torch 2.4.0,
# which would fight the torch==2.12.0 already pinned in
# requirements.txt. A modern PyPI torch wheel already bundles its own
# CUDA runtime libraries (see the OCR_DEVICE comment in .env.example),
# so all this image needs to provide is the CUDA/cuDNN userspace libs
# other native deps (onnxruntime, paddlepaddle) expect to find, plus
# whatever the NVIDIA Container Toolkit injects from the host driver
# at `docker run`/`docker compose up` time.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OCR_DEVICE=cuda

WORKDIR /app

# Ubuntu 22.04 ships Python 3.10; deadsnakes provides 3.12 to match
# the versions this pipeline was built/tested against.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        curl \
        libgl1 \
        libglib2.0-0 \
        libpq5 \
        tesseract-ocr \
        build-essential \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
    && python3.12 -m ensurepip --upgrade \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python3.12 -m pip install --no-cache-dir -r requirements.txt
RUN python3.12 -m pip install --no-cache-dir doclayout-yolo==0.0.4 "python-multipart>=0.0.9" "psycopg[binary]==3.3.4"


COPY . .

RUN mkdir -p uploads output cache models/tessdata

# Tesseract OCR language data for the languages RapidOCR/EasyOCR have
# no usable model for (see pipeline/ocr/tesseract_engine.py --
# TesseractOCREngine reads from models/tessdata via --tessdata-dir,
# never from the system tesseract-ocr package's own tessdata folder,
# so this is the only tessdata location that matters at runtime). Not
# in the models/ directory locally (gitignored, like the YOLO/UTRNet
# weights) and not part of the tesseract-ocr apt package above, so
# each is fetched directly here rather than assuming it is already
# present in the build context.
#
# curl with --retry, not ADD <url> -- ADD has no retry logic at all,
# and a transient drop fetching from raw.githubusercontent.com
# (observed directly: two of these nine failed with "unexpected EOF"
# on an otherwise fine connection) fails the entire build outright.
RUN set -eux; \
    for lang in guj mal urd pan ben asm ori tel kan tam; do \
        curl -fSL --retry 5 --retry-delay 3 --retry-all-errors \
            -o "models/tessdata/${lang}.traineddata" \
            "https://github.com/tesseract-ocr/tessdata_best/raw/main/${lang}.traineddata"; \
    done

EXPOSE 8000

# 1 worker for GPU deployment -- each worker loads DocLayout-YOLO/UTRNet
# onto the GPU (6-8 GB). On a 16GB GPU, running 2 workers causes CUDA OOM.
CMD ["gunicorn", "backend.main:app", "-k", "uvicorn.workers.UvicornWorker", "--workers", "1", "--bind", "0.0.0.0:8000"]

