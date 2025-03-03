import joblib
import os
import requests
from ml_model.config import SGD_MODEL_PATH, VECTORIZER_PATH, SGD_MODEL_URL, VECTORIZER_URL

def download_file(url, filename):
    """Scarica un file da un URL solo se non è già presente localmente."""
    if not os.path.exists(filename):
        response = requests.get(url)
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"✅ Scaricato {filename}")
    else:
        print(f"⚠️ {filename} già presente, uso quello locale.")

# Scarichiamo e carichiamo il modello
download_file(SGD_MODEL_URL, SGD_MODEL_PATH)
download_file(VECTORIZER_URL, VECTORIZER_PATH)

sgd_model = joblib.load(SGD_MODEL_PATH)
vectorizer_sgd = joblib.load(VECTORIZER_PATH)

def predict(description):
    """Predice il conto contabile dato un testo descrittivo."""
    features = vectorizer_sgd.transform([description])
    return sgd_model.predict(features)[0]

def update_model(description, correct_account):
    """Aggiorna il modello con un nuovo esempio."""
    features = vectorizer_sgd.transform([description])
    sgd_model.partial_fit(features, [correct_account])  # Usa solo le classi pre-addestrate
    joblib.dump(sgd_model, SGD_MODEL_PATH)
