from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from ml_model.model_handler import predict, update_model, get_model_stats

from fastapi.responses import FileResponse
import os
import datetime
import json
import requests
from ml_model.config import MODEL_PATH, VECTORIZER_PATH, MODEL_URL, VECTORIZER_URL

CORRECTIONS_FILE = "correzioni.json"

def save_correction(description, amount, correct_account):
    """Salva la correzione in un file JSON"""
    correction = {
        "Date": datetime.datetime.today().strftime("%Y-%m-%d"),
        "Description": description,
        "Importo": amount,
        "Target_Account": correct_account
    }

    try:
        # Legge il file se esiste, altrimenti inizializza una lista vuota
        if os.path.exists(CORRECTIONS_FILE):
            with open(CORRECTIONS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f) if f.read().strip() else []
        else:
            data = []

        # Aggiunge la nuova correzione
        data.append(correction)

        # Salva il file aggiornato
        with open(CORRECTIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        # Log minimale con il numero di correzioni totali
        print(f"✅ Correzione salvata, totale correzioni: {len(data)}")

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
            headers={
                "Content-Disposition": "attachment; filename=correzioni.json",
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    else:
        raise HTTPException(status_code=404, detail="Il file correzioni.json non esiste")

@app.get("/force-download/models")
def force_download_models():
    """Forza il download e la sovrascrittura del modello e del vectorizer"""
    try:
        # Scarica e sovrascrive il modello
        model_response = requests.get(MODEL_URL)
        if model_response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                f.write(model_response.content)
            print("✅ Modello aggiornato con successo.")
        else:
            raise HTTPException(status_code=500, detail=f"Errore nel download del modello: {model_response.status_code}")

        # Scarica e sovrascrive il vettorizzatore
        vectorizer_response = requests.get(VECTORIZER_URL)
        if vectorizer_response.status_code == 200:
            with open(VECTORIZER_PATH, "wb") as f:
                f.write(vectorizer_response.content)
            print("✅ Vettorizzatore aggiornato con successo.")
        else:
            raise HTTPException(status_code=500, detail=f"Errore nel download del vettorizzatore: {vectorizer_response.status_code}")

        return {"message": "Modello e vectorizer aggiornati con successo"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
def get_stats():
    """Genera un file di testo con le statistiche del modello e delle correzioni."""
    try:
        # Ottieni le informazioni sul modello
        model_stats = get_model_stats()

        # Conta le correzioni presenti
        if os.path.exists(CORRECTIONS_FILE):
            with open(CORRECTIONS_FILE, "r", encoding="utf-8") as f:
                corrections = json.load(f)
                num_corrections = len(corrections)
        else:
            num_corrections = 0

        # Crea il contenuto del file di statistiche
        stats_content = (
            f"📊 STATISTICHE SERVER\n\n"
            f"🧠 Modello:\n"
            f" - Nome: {model_stats['Model Name']}\n"
            f" - Dimensione: {model_stats['Model Size (bytes)']} bytes\n"
            f" - Ultima modifica: {model_stats['Model Last Modified']}\n\n"
            f"📚 Vettorizzatore:\n"
            f" - Nome: {model_stats['Vectorizer Name']}\n"
            f" - Dimensione: {model_stats['Vectorizer Size (bytes)']} bytes\n"
            f" - Ultima modifica: {model_stats['Vectorizer Last Modified']}\n\n"
            f"📂 Correzioni:\n"
            f" - Numero totale di correzioni: {num_corrections}\n"
        )

        # Salva il file temporaneo
        stats_file = "server_stats.txt"
        with open(stats_file, "w", encoding="utf-8") as f:
            f.write(stats_content)

        return FileResponse(
            path=stats_file,
            filename="server_stats.txt",
            media_type="text/plain",
            headers={"Content-Disposition": "attachment; filename=server_stats.txt"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore nella generazione delle statistiche: {e}")
