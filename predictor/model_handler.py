import joblib
import os
import datetime
import requests
import json
from fastapi import HTTPException
from fastapi.responses import FileResponse
from predictor.config import (
    get_model_path,
    get_vectorizer_path,
    get_corrections_path,
    get_stats_path,
    get_accounts_path,
    get_model_name,
    get_vectorizer_name,
    get_corrections_name,
    get_accounts_name
)
from predictor.storage import lazy_download_file

def lazy_load_file(filepath, filename, user_id):
    if not os.path.exists(filepath):
        lazy_download_file(user_id, filename, filepath)

def predict_category(user_id, text):
    model_path = get_model_path(user_id)
    vectorizer_path = get_vectorizer_path(user_id)
    lazy_load_file(model_path, get_model_name(user_id), user_id)
    lazy_load_file(vectorizer_path, get_vectorizer_name(user_id), user_id)
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    features = vectorizer.transform([text])
    prediction = model.predict(features)[0]
    return prediction

def register_feedback(user_id, json_payload):
    try:
        data = json.loads(json_payload)
        text = data["text"]
        label = data["label"]
    except Exception:
        raise HTTPException(status_code=400, detail="Formato del feedback non valido")

    accounts_path = get_accounts_path(user_id)
    lazy_load_file(accounts_path, get_accounts_name(user_id), user_id)
    try:
        with open(accounts_path, "r", encoding="utf-8") as f:
            accounts = json.load(f)
            valid_labels = list(accounts.keys())
    except Exception:
        raise HTTPException(status_code=500, detail="Errore nella lettura degli account disponibili")

    if label not in valid_labels:
        raise HTTPException(status_code=400, detail=f"Conto '{label}' non riconosciuto tra i conti validi.")

    model_path = get_model_path(user_id)
    vectorizer_path = get_vectorizer_path(user_id)
    lazy_load_file(model_path, get_model_name(user_id), user_id)
    lazy_load_file(vectorizer_path, get_vectorizer_name(user_id), user_id)
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    features = vectorizer.transform([text])
    model.partial_fit(features, [label])
    joblib.dump(model, model_path)
    print("✅ Modello aggiornato con il nuovo feedback")

    correzione = {
        "timestamp": datetime.datetime.now().isoformat(),
        "text": text,
        "label": label
    }

    corrections_path = get_corrections_path(user_id)
    try:
        if os.path.exists(corrections_path):
            with open(corrections_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        else:
            existing = []

        existing.append(correzione)
        with open(corrections_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)
    except Exception as e:
        print(f"⚠️ Errore salvataggio correzione: {e}")

    return {"status": "ok"}

def get_model_stats(user_id):
    model_path = get_model_path(user_id)
    vectorizer_path = get_vectorizer_path(user_id)
    corrections_path = get_corrections_path(user_id)
    stats_path = get_stats_path(user_id)

    lazy_load_file(model_path, get_model_name(user_id), user_id)
    lazy_load_file(vectorizer_path, get_vectorizer_name(user_id), user_id)
    lazy_load_file(corrections_path, get_corrections_name(user_id), user_id)

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

    num_corrections = 0
    last_corrections = []
    if os.path.exists(corrections_path):
        try:
            with open(corrections_path, "r", encoding="utf-8") as f:
                corrections = json.load(f)
                num_corrections = len(corrections) if isinstance(corrections, list) else 0
                last_corrections = corrections[-5:]
        except json.JSONDecodeError:
            print(f"⚠️ Warning: Il file {corrections_path} non è un JSON valido. Ignorato.")
            num_corrections = 0

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

    if num_corrections > 0:
        stats_content += "\n📜 Ultime correzioni:\n"
        for c in last_corrections:
            stats_content += f" - {json.dumps(c, ensure_ascii=False)}\n"

    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(stats_content)

    return FileResponse(
        path=stats_path,
        filename=os.path.basename(stats_path),
        media_type="text/plain"
    )

def download_model(user_id):
    model_path = get_model_path(user_id)
    lazy_load_file(model_path, get_model_name(user_id), user_id)
    return FileResponse(model_path, filename="model.pkl")

def download_vectorizer(user_id):
    vectorizer_path = get_vectorizer_path(user_id)
    lazy_load_file(vectorizer_path, get_vectorizer_name(user_id), user_id)
    return FileResponse(vectorizer_path, filename="vectorizer.pkl")

def download_corrections(user_id):
    corrections_path = get_corrections_path(user_id)
    lazy_load_file(corrections_path, get_corrections_name(user_id), user_id)
    if os.path.exists(corrections_path):
        return FileResponse(corrections_path, filename="corrections.json")
    else:
        return {"corrections": []}

def download_labels(user_id):
    labels_path = get_accounts_path(user_id)
    lazy_load_file(labels_path, get_accounts_name(user_id), user_id)
    if os.path.exists(labels_path):
        return FileResponse(labels_path, filename="accounts.json")
    else:
        return {"accounts": []}