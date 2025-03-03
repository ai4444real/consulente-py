from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from ml_model.model_handler import predict, update_model

from fastapi.responses import FileResponse
import os
from ml_model.config import MODEL_PATH, VECTORIZER_PATH

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
        update_model(transaction.description, transaction.correctAccount)
        return {"message": "Correzione registrata con successo!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#@app.get("/download/model")
#def download_model():
#    """Endpoint per scaricare il modello allenato."""
#    if os.path.exists(MODEL_PATH):
#        return FileResponse(MODEL_PATH, filename="modello_sgd.pkl", media_type="application/octet-stream")
#    else:
#        raise HTTPException(status_code=404, detail="Modello non trovato")

@app.get("/download/model")
def download_model():
    """Endpoint per scaricare il modello allenato, con debug."""
    if os.path.exists(MODEL_PATH):
        file_size = os.path.getsize(MODEL_PATH)  # Otteniamo la dimensione del file
        return {
            "status": "ok",
            "message": "Modello trovato",
            "file_path": MODEL_PATH,
            "file_size": file_size
        }
    else:
        return {
            "status": "error",
            "message": "Modello non trovato",
            "file_path": MODEL_PATH
        }


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
