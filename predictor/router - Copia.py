from fastapi import APIRouter, UploadFile, File
from fastapi.responses import FileResponse
from predictor.model_handler import (
    predict_category,
    register_feedback,
    get_model_stats,
    download_model,
    download_vectorizer,
    download_corrections
)

router = APIRouter(prefix="/predictor")

@router.post("/predict/{user_id}")
async def predict(user_id: str, file: UploadFile = File(...)):
    content = await file.read()
    prediction = predict_category(user_id, content.decode("utf-8"))
    return {"prediction": prediction}

@router.post("/feedback/{user_id}")
async def feedback(user_id: str, file: UploadFile = File(...)):
    content = await file.read()
    return register_feedback(user_id, content.decode("utf-8"))

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
