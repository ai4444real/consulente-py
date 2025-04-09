from fastapi import FastAPI, UploadFile, File
import subprocess

app = FastAPI()

@app.post("/ocr")
async def run_ocr(file: UploadFile = File(...)):
    # Salva il file PDF temporaneamente
    with open("temp.pdf", "wb") as f:
        f.write(await file.read())
    
    # Esegui lo script e passa il file come argomento
    try:
        output = subprocess.check_output(["python", "ocr_extract.py", "temp.pdf"])
        return {"text": output.decode("utf-8")}
    except subprocess.CalledProcessError as e:
        return {"error": e.output.decode("utf-8")}
