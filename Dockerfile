FROM python:3.11-slim

# Installa strumenti OCR (utili per pdf_reader)
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-ita \
    && apt-get clean

# Crea directory principale
WORKDIR /app

# Copia tutto il codice (includendo predictor, pdf_reader, main.py, ecc.)
COPY . /app

# Installa le dipendenze
RUN pip install --no-cache-dir -r requirements.txt

# Rendi visibili i moduli ai percorsi
ENV PYTHONPATH=/app

# Espone la porta usata da uvicorn
EXPOSE 8000

# Avvia l'applicazione unificata
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

