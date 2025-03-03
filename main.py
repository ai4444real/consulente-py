from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import requests

# URL dei file su Supabase (SOSTITUISCI con gli URL reali)
SGD_MODEL_URL = "https://zulsyfmxuczxfkygphkb.supabase.co/storage/v1/object/public/models//modello_sgd.pkl"
VECTORIZER_URL = "https://zulsyfmxuczxfkygphkb.supabase.co/storage/v1/object/public/models//vectorizer_sgd.pkl"

# Scarica i file da Supabase
def download_file(url, filename):
    response = requests.get(url)
    with open(filename, "wb") as f:
        f.write(response.content)

download_file(SGD_MODEL_URL, "modello_sgd.pkl")
download_file(VECTORIZER_URL, "vectorizer_sgd.pkl")

# Carichiamo il modello e il vectorizer
sgd_model = joblib.load("modello_sgd.pkl")
vectorizer_sgd = joblib.load("vectorizer_sgd.pkl")

print("✅ Modello SGDClassifier caricato con successo!")

# Inizializziamo FastAPI
app = FastAPI()

# Definizione della struttura dei dati in ingresso
class Transaction(BaseModel):
    description: str
    amount: float

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

# Endpoint di test
@app.get("/")
def home():
    return {"message": "API di Predizione Contabile attiva con SGDClassifier"}
