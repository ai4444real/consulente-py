
import json
from fastapi import APIRouter, Body
from pydantic import BaseModel
from predictor.model_handler import (
    predict_category,
    register_feedback,
    get_model_stats,
    download_model,
    download_vectorizer,
    download_corrections,
    download_labels
)

router = APIRouter(prefix="/predictor")

class PredictionInput(BaseModel):
    text: str

@router.post("/predict/{user_id}")
def predict(user_id: str, input: PredictionInput = Body(...)):
    prediction = predict_category(user_id, input.text)
    return {"predictedAccount": prediction}

class FeedbackInput(BaseModel):
    text: str
    label: str

@router.post("/feedback/{user_id}")
def feedback(user_id: str, input: FeedbackInput = Body(...)):
    json_payload = json.dumps(input.model_dump())
    return register_feedback(user_id, json_payload)

@router.get("/stats/{user_id}")
def stats(user_id: str):
    return get_model_stats(user_id)

@router.get("/download/model/{user_id}")
def get_model(user_id: str):
    return download_model(user_id)

@router.get("/download/vectorizer/{user_id}")
def get_vectorizer(user_id: str):
    return download_vectorizer(user_id)

@router.get("/download/corrections/{user_id}")
def get_corrections(user_id: str):
    return download_corrections(user_id)

@router.get("/download/labels/{user_id}")
def get_labels(user_id: str):
    return download_labels(user_id)
