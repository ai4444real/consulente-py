from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from ml_model.model_handler import predict, update_model

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

@app.get("/")
def home():
    return {"message": "API di Predizione Contabile attiva con SGDClassifier"}
