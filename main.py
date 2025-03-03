from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import requests

# URL dei file su Supabase (SOSTITUISCI con gli URL reali)
SGD_MODEL_URL = "https://zulsyfmxuczxfkygphkb.supabase.co/storage/v1/object/public/models//modello_sgd.pkl"
VECTORIZER_URL = "https://zulsyfmxuczxfkygphkb.supabase.co/storage/v1/object/public/models//vectorizer_sgd.pkl"

# Scarica i file da Supabase
import os

def download_file(url, filename):
    if not os.path.exists(filename):  # Scarica solo se il file non esiste
        response = requests.get(url)
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"✅ Scaricato {filename} da Supabase.")
    else:
        print(f"⚠️ {filename} già presente, uso quello locale.")

download_file(SGD_MODEL_URL, "modello_sgd.pkl")
download_file(VECTORIZER_URL, "vectorizer_sgd.pkl")

# Carichiamo il modello e il vectorizer
sgd_model = joblib.load("modello_sgd.pkl")
vectorizer_sgd = joblib.load("vectorizer_sgd.pkl")

print("✅ Modello SGDClassifier caricato con successo!")

# Inizializziamo FastAPI
app = FastAPI()

# Abilitiamo i CORS per consentire richieste dall'interfaccia web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Puoi restringere ai tuoi domini se necessario
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Definizione della struttura dei dati in ingresso
class Transaction(BaseModel):
    description: str
    amount: float
    correctAccount: str = None 

# Endpoint per predire il conto contabile
@app.post("/predict")
def predict(transaction: Transaction):
    try:
        # Convertiamo la descrizione in numeri con il vectorizer
        features = vectorizer_sgd.transform([transaction.description])

        # Prediciamo il conto contabile
        predicted_account = sgd_model.predict(features)[0]

        return {
            "description": transaction.description,
            "amount": transaction.amount,
            "predictedAccount": predicted_account
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


import json
CLASS_FILE = "classes.json"  # File per memorizzare le classi viste dal modello

# Funzione per caricare le classi dal file
def load_classes():
    try:
        with open(CLASS_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return list(sgd_model.classes_)  # Se il file non esiste, usiamo le classi attuali

# Funzione per salvare le classi aggiornate
def save_classes(classes):
    with open(CLASS_FILE, "w") as f:
        json.dump(classes, f)

@app.post("/feedback")
def feedback(transaction: Transaction):
    try:
        # Convertiamo la descrizione in una rappresentazione numerica
        features = vectorizer_sgd.transform([transaction.description])

        # Controlliamo se il conto corretto è già nel modello
        current_classes = set(map(str, sgd_model.classes_))
        print(f"⚠️ Classi attuali nel modello: {current_classes}")
        print(f"✅ Nuova classe da aggiungere: {transaction.correctAccount}")

        # Se la nuova classe è già conosciuta, aggiorniamo senza toccare le classi
        if str(transaction.correctAccount) in current_classes:
            print("🔄 Aggiornamento senza modificare le classi esistenti.")
            sgd_model.partial_fit(features, [transaction.correctAccount])  # Senza specificare le classi
        else:
            # Caso raro: se una classe nuova viene aggiunta, dobbiamo riallenare con TUTTE le classi
            print("⚠️ Nuova classe rilevata! Riaddestriamo il modello con tutte le classi conosciute.")
            new_classes = sorted(current_classes | {str(transaction.correctAccount)})
            sgd_model.partial_fit(features, [transaction.correctAccount], classes=new_classes)

        # Salviamo il modello aggiornato
        joblib.dump(sgd_model, "modello_sgd.pkl")

        return {"message": "Correzione registrata con successo!"}
    except Exception as e:
        print(f"❌ Errore durante l'aggiornamento: {e}")
        raise HTTPException(status_code=500, detail=str(e))



# Endpoint di test
@app.get("/")
def home():
    return {"message": "API di Predizione Contabile attiva con SGDClassifier"}
