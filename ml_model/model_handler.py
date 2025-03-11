import joblib
import os
import datetime
import requests
import json
from fastapi import HTTPException
from fastapi.responses import FileResponse
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

def predict(description, user_id):
    """Predice il conto contabile dato un testo descrittivo usando il modello dell'utente."""

    model_path = f"models/{user_id}_model_sgd.pkl"
    vectorizer_path = f"models/{user_id}_vectorizer_sgd.pkl"

    # Se il modello NON esiste, restituisce errore
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise HTTPException(status_code=404, detail=f"Modello non trovato per l'utente '{user_id}'")

    # Carica il modello e il vettorizzatore dell'utente
    sgd_model = joblib.load(model_path)
    vectorizer_sgd = joblib.load(vectorizer_path)

    # Effettua la predizione
    features = vectorizer_sgd.transform([description])
    return sgd_model.predict(features)[0]

def update_model(description, correct_account, user_id):
    """Aggiorna il modello dell'utente con un nuovo esempio e salva le modifiche."""
    
    model_path = f"models/{user_id}_model_sgd.pkl"
    vectorizer_path = f"models/{user_id}_vectorizer_sgd.pkl"

    # Se il modello NON esiste, restituisce errore
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise HTTPException(status_code=404, detail=f"Modello non trovato per l'utente '{user_id}'")

    # Carica il modello e il vettorizzatore dell'utente
    sgd_model = joblib.load(model_path)
    vectorizer_sgd = joblib.load(vectorizer_path)

    # Aggiorna il modello con il nuovo esempio
    features = vectorizer_sgd.transform([description])
    sgd_model.partial_fit(features, [correct_account])

    # Salva il modello aggiornato
    joblib.dump(sgd_model, model_path)
    print(f"✅ Modello aggiornato per {user_id} e salvato.")

def reload_model(user_id):
    """Scarica e ricarica il modello e il vettorizzatore di un utente specifico"""
    model_path = f"models/{user_id}_model_sgd.pkl"
    vectorizer_path = f"models/{user_id}_vectorizer_sgd.pkl"

    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise HTTPException(status_code=404, detail=f"Modello non trovato per l'utente '{user_id}'")

    try:
        sgd_model = joblib.load(model_path)
        vectorizer_sgd = joblib.load(vectorizer_path)
        print(f"✅ Modello e vettorizzatore ricaricati con successo per {user_id}.")
        return {"message": f"Modello e vettorizzatore ricaricati per {user_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def generate_user_stats(user_id):
    """Restituisce informazioni sul modello, vettorizzatore e correzioni per un utente specifico."""
    model_path = f"models/{user_id}_model_sgd.pkl"
    vectorizer_path = f"models/{user_id}_vectorizer_sgd.pkl"
    corrections_file = f"corrections/{user_id}_corrections.json"

    # Controlla se il modello e il vettorizzatore esistono
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise HTTPException(status_code=404, detail=f"Modello non trovato per l'utente '{user_id}'")

    # Ottiene dimensione e ultima modifica dei file
    model_size = os.path.getsize(model_path) if os.path.exists(model_path) else 0
    vectorizer_size = os.path.getsize(vectorizer_path) if os.path.exists(vectorizer_path) else 0

    model_last_modified = (
        datetime.datetime.fromtimestamp(os.path.getmtime(model_path)).strftime('%Y-%m-%d %H:%M:%S')
        if os.path.exists(model_path) else "Non disponibile"
    )
    vectorizer_last_modified = (
        datetime.datetime.fromtimestamp(os.path.getmtime(vectorizer_path)).strftime('%Y-%m-%d %H:%M:%S')
        if os.path.exists(vectorizer_path) else "Non disponibile"
    )

    # Controlla le correzioni
    num_corrections = 0
    last_corrections = []
    if os.path.exists(corrections_file):
        try:
            with open(corrections_file, "r", encoding="utf-8") as f:
                corrections = json.load(f)
                num_corrections = len(corrections) if isinstance(corrections, list) else 0
                last_corrections = corrections[-5:]  # Ultime 5 correzioni
        except json.JSONDecodeError:
            print(f"⚠️ Warning: Il file {corrections_file} non è un JSON valido. Ignorato.")
            num_corrections = 0

    # Crea il contenuto del file di statistiche
    stats_content = (
        f"📊 STATISTICHE PER {user_id}\n\n"
        f"🧠 Modello:\n"
        f" - Nome: {os.path.basename(model_path)}\n"
        f" - Dimensione: {model_size} bytes\n"
        f" - Ultima modifica: {model_last_modified}\n\n"
        f"📚 Vettorizzatore:\n"
        f" - Nome: {os.path.basename(vectorizer_path)}\n"
        f" - Dimensione: {vectorizer_size} bytes\n"
        f" - Ultima modifica: {vectorizer_last_modified}\n\n"
        f"📂 Correzioni:\n"
        f" - Numero totale di correzioni: {num_corrections}\n"
    )

    # Se ci sono correzioni, aggiungile al file
    if num_corrections > 0:
        stats_content += "\n📜 Ultime correzioni:\n"
        for correction in last_corrections:
            stats_content += (
                f" - Data: {correction.get('Date', 'N/A')}, "
                f"Descrizione: {correction.get('Description', 'N/A')}, "
                f"Importo: {correction.get('Importo', 'N/A')}, "
                f"Conto: {correction.get('Target_Account', 'N/A')}\n"
            )

    # Salva il file temporaneo delle statistiche
    stats_file = f"stats/{user_id}_stats.txt"
    os.makedirs("stats", exist_ok=True)

    with open(stats_file, "w", encoding="utf-8") as f:
        f.write(stats_content)

    return FileResponse(
        path=stats_file,
        filename=f"{user_id}_stats.txt",
        media_type="text/plain"
    )

