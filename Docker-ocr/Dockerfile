FROM python:3.11-slim

# Installa dipendenze base
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-ita \
    poppler-utils \
    && apt-get clean

# Crea cartella per l'app
WORKDIR /app
COPY . .

# Installa le dipendenze Python
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "ocr_extract.py"]
