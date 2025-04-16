from fastapi import FastAPI
from predictor.router import router as predictor_router
from pdf_reader.router import router as pdf_reader_router

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Aggiungi questo blocco dopo aver creato l'app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in prod potresti limitare a ['https://tuodominio.com']
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Includi entrambi i router con i rispettivi prefix
app.include_router(predictor_router)
app.include_router(pdf_reader_router)

