from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from ml_model.model_handler import predict, update_model, get_model_stats, reload_model

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
def force_reload_models():
    """Forza il download e la ricarica del modello e del vectorizer"""
    return reload_model()

@app.get("/stats")
def get_stats():
    """Genera un file di testo con le statistiche del modello e delle correzioni."""
    try:
        # Ottieni le informazioni sul modello
        model_stats = get_model_stats()

        # Conta le correzioni presenti in modo robusto
        num_corrections = 0
        if os.path.exists(CORRECTIONS_FILE):
            try:
                with open(CORRECTIONS_FILE, "r", encoding="utf-8") as f:
                    content = f.read().strip()  # Rimuove spazi vuoti
                    corrections = json.loads(content) if content else []
                    num_corrections = len(corrections) if isinstance(corrections, list) else 0
            except json.JSONDecodeError:
                print(f"⚠️ Warning: Il file {CORRECTIONS_FILE} non è un JSON valido. Ignorato.")
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

        # Se ci sono correzioni, aggiungile al file
        if num_corrections > 0:
            stats_content += "\n📜 Dettaglio delle ultime correzioni:\n"
            for correction in corrections[-5:]:  # Ultime 5 correzioni
                stats_content += (
                    f" - Data: {correction.get('Date', 'N/A')}, "
                    f"Descrizione: {correction.get('Description', 'N/A')}, "
                    f"Importo: {correction.get('Importo', 'N/A')}, "
                    f"Conto: {correction.get('Target_Account', 'N/A')}\n"
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

from fastapi.responses import JSONResponse

@app.get("/accounts/{user_id}")
def get_accounts():
    """Restituisce la lista di account di default dal server."""
    path = "accounts/{user_id}_accounts.json"

    if not os.path.exists(path):
        return JSONResponse(content={}, status_code=404)

    with open(path, "r", encoding="utf-8") as f:
        accounts = json.load(f)

    return JSONResponse(content=accounts)
