FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p uploads output cache models/tessdata

# Tesseract OCR language data for the languages RapidOCR/EasyOCR have
# no usable model for (see pipeline/ocr/tesseract_engine.py and
# PipelineService._get_ocr_engine_for_language -- this now includes
# Telugu/Kannada too: RapidOCR ships real recognition models for both,
# but a measured comparison found its shared detector fails outright
# on some Telugu conjuncts and badly garbles Kannada, while Tesseract
# reads the same crops correctly). Not in the models/ directory
# locally (gitignored, like the YOLO layout weights) and not part of
# the tesseract-ocr apt package above, so each is fetched directly
# here rather than assuming it is already present in the build
# context.
ADD https://github.com/tesseract-ocr/tessdata_best/raw/main/guj.traineddata \
    models/tessdata/guj.traineddata
ADD https://github.com/tesseract-ocr/tessdata_best/raw/main/mal.traineddata \
    models/tessdata/mal.traineddata
ADD https://github.com/tesseract-ocr/tessdata_best/raw/main/urd.traineddata \
    models/tessdata/urd.traineddata
ADD https://github.com/tesseract-ocr/tessdata_best/raw/main/pan.traineddata \
    models/tessdata/pan.traineddata
ADD https://github.com/tesseract-ocr/tessdata_best/raw/main/ben.traineddata \
    models/tessdata/ben.traineddata
ADD https://github.com/tesseract-ocr/tessdata_best/raw/main/asm.traineddata \
    models/tessdata/asm.traineddata
ADD https://github.com/tesseract-ocr/tessdata_best/raw/main/ori.traineddata \
    models/tessdata/ori.traineddata
ADD https://github.com/tesseract-ocr/tessdata_best/raw/main/tel.traineddata \
    models/tessdata/tel.traineddata
ADD https://github.com/tesseract-ocr/tessdata_best/raw/main/kan.traineddata \
    models/tessdata/kan.traineddata

EXPOSE 8000

CMD ["gunicorn", "backend.main:app", "-k", "uvicorn.workers.UvicornWorker", "--workers", "4", "--bind", "0.0.0.0:8000"]
