import joblib
import os
import datetime
import requests
from fastapi import HTTPException
from ml_model.config import MODEL_PATH, VECTORIZER_PATH, MODEL_URL, VECTORIZER_URL

def download_file(url, filepath):
    """Scarica un file dal URL se non è già presente."""
    if not os.path.exists(filepath):
        print(f"📥 Scaricamento {filepath} da {url}...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(filepath, "wb") as f:
                f.write(response.content)
            print(f"✅ {filepath} scaricato e salvato.")
        else:
            raise HTTPException(status_code=500, detail=f"Errore nel download di {filepath}: {response.status_code}")

# Assicuriamoci che i file esistano prima di caricarli
download_file(MODEL_URL, MODEL_PATH)
download_file(VECTORIZER_URL, VECTORIZER_PATH)

# Carichiamo il modello una sola volta
sgd_model = joblib.load(MODEL_PATH)
vectorizer_sgd = joblib.load(VECTORIZER_PATH)
print("✅ Modello e vettorizzatore caricati con successo.")

def predict(description):
    """Predice il conto contabile dato un testo descrittivo usando il modello 'default'."""
    if sgd_model is None or vectorizer_sgd is None:
        raise HTTPException(status_code=500, detail="Modello non caricato correttamente.")

    features = vectorizer_sgd.transform([description])
    return sgd_model.predict(features)[0]

def update_model(description, correct_account):
    """Aggiorna il modello con un nuovo esempio e salva le modifiche."""
    if sgd_model is None or vectorizer_sgd is None:
        raise HTTPException(status_code=500, detail="Modello non caricato correttamente.")

    features = vectorizer_sgd.transform([description])
    sgd_model.partial_fit(features, [correct_account])

    # Salviamo il modello aggiornato
    joblib.dump(sgd_model, MODEL_PATH)
    print("✅ Modello aggiornato e salvato.")

def reload_model():
    """Ricarica il modello e il vettorizzatore"""
    global sgd_model, vectorizer_sgd

    try:
        sgd_model = joblib.load(MODEL_PATH)
        vectorizer_sgd = joblib.load(VECTORIZER_PATH)
        print("✅ Modello e vettorizzatore ricaricati con successo.")
        return {"message": "Modello e vettorizzatore ricaricati con successo"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_model_stats():
    """Restituisce informazioni sul modello e il vectorizer"""
    try:
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

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
