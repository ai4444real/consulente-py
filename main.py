from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from ml_model.model_handler import predict, update_model

from fastapi.responses import FileResponse
import os
import datetime
import json
from ml_model.config import MODEL_PATH, VECTORIZER_PATH

CORRECTIONS_FILE = "correzioni.json"

def save_correction(description, amount, correct_account):
    """Salva la correzione in un file JSON"""
    correction = {
        "Date": datetime.datetime.today().strftime("%Y-%m-%d") if datetime.datetime.today() else "1970-01-01",
        "Description": description,
        "Importo": amount,
        "Target_Account": correct_account
    }

    try:
        # Se il file esiste, leggiamo il contenuto e aggiungiamo la nuova correzione
        if os.path.exists(CORRECTIONS_FILE):
            with open(CORRECTIONS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = []

        # Aggiungiamo la nuova correzione
        data.append(correction)

        # Salviamo di nuovo il file
        with open(CORRECTIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        # ✅ Stampa il contenuto del file dopo il salvataggio
        with open(CORRECTIONS_FILE, "r", encoding="utf-8") as f:
            saved_data = f.read()
        
            print(f"✅ Correzione salvata in '{CORRECTIONS_FILE}', numero di correzioni totali: {len(data)}")
            print(f"📄 Contenuto attuale di {CORRECTIONS_FILE}:\n{saved_data}")
    except Exception as e:
        print(f"❌ Errore nel salvataggio della correzione: {e}")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Transaction(BaseModel):
    description: str
    amount: float
    correctAccount: str = None 

@app.post("/predict")
def predict_transaction(transaction: Transaction):
    try:
        predicted_account = predict(transaction.description)
        return {
            "description": transaction.description,
            "amount": transaction.amount,
            "predictedAccount": predicted_account
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
def feedback(transaction: Transaction):
    try:
        save_correction(transaction.description, transaction.amount, transaction.correctAccount)
        update_model(transaction.description, transaction.correctAccount)
        return {"message": "Correzione registrata con successo!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from fastapi.responses import FileResponse

@app.get("/download/model")
def download_model():
    """Forza il download del modello"""
    if os.path.exists(MODEL_PATH):
        return FileResponse(
            path=MODEL_PATH,
            filename="modello_sgd.pkl",
            media_type="application/octet-stream",
            headers={"Content-Disposition": "attachment; filename=modello_sgd.pkl"}
        )
    else:
        raise HTTPException(status_code=404, detail="Modello non trovato")



@app.get("/download/vectorizer")
def download_vectorizer():
    """Endpoint per scaricare il vettorizzatore."""
    if os.path.exists(VECTORIZER_PATH):
        return FileResponse(VECTORIZER_PATH, filename="vectorizer_sgd.pkl", media_type="application/octet-stream")
    else:
        raise HTTPException(status_code=404, detail="Vectorizer non trovato")

@app.get("/")
def home():
    return {"message": "API di Predizione Contabile attiva con SGDClassifier"}

@app.get("/download/corrections")
def download_corrections():
    """Permette di scaricare il file delle correzioni."""
    if os.path.exists(CORRECTIONS_FILE):
        return FileResponse(
            path=CORRECTIONS_FILE,
            filename="correzioni.json",
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=correzioni.json"}
        )
    else:
        raise HTTPException(status_code=404, detail="Il file correzioni.json non esiste")
