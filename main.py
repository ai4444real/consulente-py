from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from ml_model.model_handler import predict, update_model, generate_user_stats, reload_model

from fastapi.responses import FileResponse
import os
import datetime
import json
import requests
from ml_model.config import MODEL_PATH, VECTORIZER_PATH, MODEL_URL, VECTORIZER_URL

CORRECTIONS_FILE = "corrections.json"

def save_correction(description, amount, correct_account, user_id):
    """Salva la correzione nel file JSON specifico dell'utente."""
    correction = {
        "Date": datetime.datetime.today().strftime("%Y-%m-%d"),
        "Description": description,
        "Importo": amount,
        "Target_Account": correct_account
    }

    corrections_file = f"corrections/{user_id}_corrections.json"

    try:
        # Verifica se il file esiste e non è vuoto
        if os.path.exists(corrections_file):
            with open(corrections_file, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)  # Prova a caricare il JSON
                except json.JSONDecodeError:
                    print(f"⚠️ Il file {corrections_file} era corrotto o vuoto. Verrà ricreato.")
                    data = []
        else:
            data = []  # Se il file non esiste, inizializza una lista vuota

        # Aggiunge la nuova correzione
        data.append(correction)

        # Salva il file aggiornato
        with open(corrections_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        print(f"✅ Correzione salvata per {user_id}, totale correzioni: {len(data)}")

    except Exception as e:
        print(f"❌ Errore nel salvataggio della correzione per {user_id}: {e}")


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

@app.post("/predict/{user_id}")
def predict_transaction(user_id: str, transaction: Transaction):
    try:
        predicted_account = predict(transaction.description, user_id)  # PASSIAMO user_id
        return {
            "description": transaction.description,
            "amount": transaction.amount,
            "predictedAccount": predicted_account
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback/{user_id}")
def feedback(user_id: str, transaction: Transaction):
    try:
        save_correction(transaction.description, transaction.amount, transaction.correctAccount, user_id)  
        update_model(transaction.description, transaction.correctAccount, user_id)  
        return {"message": "Correzione registrata con successo!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from fastapi.responses import FileResponse

@app.get("/download/model/{user_id}")
def download_model(user_id: str):
    """Scarica il modello specifico dell'utente."""
    model_path = f"models/{user_id}_model_sgd.pkl"

    if os.path.exists(model_path):
        return FileResponse(
            path=model_path,
            filename=f"{user_id}_model_sgd.pkl",
            media_type="application/octet-stream"
        )
    else:
        raise HTTPException(status_code=404, detail=f"Modello non trovato per l'utente '{user_id}'")


@app.get("/download/vectorizer/{user_id}")
def download_vectorizer(user_id: str):
    """Scarica il vettorizzatore specifico dell'utente."""
    vectorizer_path = f"models/{user_id}_vectorizer_sgd.pkl"

    if os.path.exists(vectorizer_path):
        return FileResponse(
            path=vectorizer_path,
            filename=f"{user_id}_vectorizer_sgd.pkl",
            media_type="application/octet-stream"
        )
    else:
        raise HTTPException(status_code=404, detail=f"Vectorizer non trovato per l'utente '{user_id}'")

@app.get("/")
def home():
    return {"message": "API di Predizione Contabile attiva con SGDClassifier"}

@app.get("/download/corrections/{user_id}")
def download_corrections(user_id: str):
    """Scarica il file delle corrections.json specifico dell'utente."""
    corrections_file = f"corrections/{user_id}_corrections.json"

    if os.path.exists(corrections_file):
        return FileResponse(
            path=corrections_file,
            filename=f"{user_id}_corrections.json",
            media_type="application/json",
            headers={
                "Content-Disposition": f"attachment; filename={user_id}_corrections.json",
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    else:
        raise HTTPException(status_code=404, detail=f"Nessuna correzione trovata per l'utente '{user_id}'")

@app.get("/force-download/models/{user_id}")
def force_reload_models(user_id: str):
    """Forza il download e la ricarica del modello e del vectorizer per un utente specifico"""
    return reload_model(user_id)

@app.get("/stats/{user_id}")
def get_stats(user_id: str):
    """Genera un file di statistiche per un utente specifico."""
    return generate_user_stats(user_id)

from fastapi.responses import JSONResponse

@app.get("/accounts/{user_id}")
def get_accounts(user_id: str):
    """Restituisce la lista di account di un utente specifico."""
    path = f"accounts/{user_id}_accounts.json"  # CORRETTO

    if not os.path.exists(path):
        return JSONResponse(content={}, status_code=404)

    with open(path, "r", encoding="utf-8") as f:
        accounts = json.load(f)

    return JSONResponse(content=accounts)

