import joblib
import os
import datetime
import requests
from ml_model.config import MODEL_PATH, VECTORIZER_PATH, MODEL_URL, VECTORIZER_URL
from fastapi import HTTPException

CORRECTIONS_FILE = "correzioni.json"

def download_file(url, filename):
    """Scarica un file da un URL solo se non è già presente localmente."""
    if not os.path.exists(filename):
        response = requests.get(url)
        with open(filename, "wb") as f:
            f.write(response.content)

        file_size = os.path.getsize(filename)  # Ottieni la dimensione in byte
        last_modified = datetime.datetime.fromtimestamp(os.path.getmtime(filename)).strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"✅ Scaricato {filename} ({file_size} bytes, ultima modifica: {last_modified})")
    else:
        file_size = os.path.getsize(filename)  # Ottieni la dimensione in byte
        last_modified = datetime.datetime.fromtimestamp(os.path.getmtime(filename)).strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"⚠️ {filename} già presente ({file_size} bytes, ultima modifica: {last_modified}), uso quello locale.")

# Scarichiamo e carichiamo il modello
download_file(MODEL_URL, MODEL_PATH)
download_file(VECTORIZER_URL, VECTORIZER_PATH)

sgd_model = joblib.load(MODEL_PATH)
vectorizer_sgd = joblib.load(VECTORIZER_PATH)

def predict(description):
    """Predice il conto contabile dato un testo descrittivo."""
    features = vectorizer_sgd.transform([description])
    return sgd_model.predict(features)[0]

def update_model(description, correct_account):
    """Aggiorna il modello con un nuovo esempio."""
    features = vectorizer_sgd.transform([description])
    sgd_model.partial_fit(features, [correct_account])  # Usa solo le classi pre-addestrate
    joblib.dump(sgd_model, MODEL_PATH)

def reload_model():
    """Scarica e ricarica il modello e il vettorizzatore"""
    global sgd_model, vectorizer_sgd

    try:
        print("🔄 Scaricamento e aggiornamento del modello e del vettorizzatore...")
        
        # Scarica entrambi i file (usiamo già download_file)
        download_file(MODEL_URL, MODEL_PATH)
        download_file(VECTORIZER_URL, VECTORIZER_PATH)

        # Ricarica i file aggiornati
        sgd_model = joblib.load(MODEL_PATH)
        vectorizer_sgd = joblib.load(VECTORIZER_PATH)

        print("✅ Modello e vettorizzatore aggiornati con successo.")
        return {"message": "Modello e vectorizer aggiornati e ricaricati con successo"}

    except Exception as e:
        print(f"❌ Errore nell'aggiornamento del modello: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def get_model_stats():
    """Restituisce informazioni sul modello e il vectorizer."""
    model_size = os.path.getsize(MODEL_PATH) if os.path.exists(MODEL_PATH) else 0
    vectorizer_size = os.path.getsize(VECTORIZER_PATH) if os.path.exists(VECTORIZER_PATH) else 0

    model_last_modified = (
        datetime.datetime.fromtimestamp(os.path.getmtime(MODEL_PATH)).strftime('%Y-%m-%d %H:%M:%S')
        if os.path.exists(MODEL_PATH) else "Non disponibile"
    )
    vectorizer_last_modified = (
        datetime.datetime.fromtimestamp(os.path.getmtime(VECTORIZER_PATH)).strftime('%Y-%m-%d %H:%M:%S')
        if os.path.exists(VECTORIZER_PATH) else "Non disponibile"
    )

    return {
        "Model Name": os.path.basename(MODEL_PATH),
        "Model Size (bytes)": model_size,
        "Model Last Modified": model_last_modified,
        "Vectorizer Name": os.path.basename(VECTORIZER_PATH),
        "Vectorizer Size (bytes)": vectorizer_size,
        "Vectorizer Last Modified": vectorizer_last_modified,
    }
