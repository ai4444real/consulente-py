from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import joblib
import os
import requests

MODEL_URL = "https://zulsyfmxuczxfkygphkb.supabase.co/storage/v1/object/public/models/modello_ai.pkl"
MODEL_PATH = "modello_ai.pkl"

# Scarica il modello AI da Supabase se non esiste
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Scaricando il modello AI...")
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        print("✅ Modello scaricato!")

download_model()

# Carica il modello AI
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

app = FastAPI()

class Transaction(BaseModel):
    description: str
    amount: float

@app.post("/predict")
def predict(transaction: Transaction):
    try:
        # Convertiamo i dati in un formato che il modello può usare
        features = [transaction.description] 
        predicted_account = model.predict(features)[0]
        
        return {
            "description": transaction.description,
            "amount": transaction.amount,
            "predictedAccount": predicted_account
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
