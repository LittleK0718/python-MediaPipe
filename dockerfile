FROM python:3.10-slim

# 設定工作目錄
WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .
COPY bert_classifier.tflite .

EXPOSE 5000

CMD ["python", "server.py"]