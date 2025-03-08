# Percorsi locali
#MODEL_PATH = "models/model.pkl"
#VECTORIZER_PATH = "models/vectorizer.pkl"

# URL su Supabase (puoi sostituirli con i veri URL)
#MODEL_URL = "https://zulsyfmxuczxfkygphkb.supabase.co/storage/v1/object/public/models/modello_sgd.pkl"
#VECTORIZER_URL = "https://zulsyfmxuczxfkygphkb.supabase.co/storage/v1/object/public/models/vectorizer_sgd.pkl"

import os

# Cartella per i modelli
MODEL_DIR = "models"

# User ID di default (per ora usiamo "default", poi sarà dinamico)
USER_ID = "default"

# Nomi file con il formato `{user_id}_{type}.pkl`
MODEL_FILENAME = f"{USER_ID}_model_sgd.pkl"
VECTORIZER_FILENAME = f"{USER_ID}_vectorizer_sgd.pkl"

# Percorsi completi
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
VECTORIZER_PATH = os.path.join(MODEL_DIR, VECTORIZER_FILENAME)

# URL per il download (già con il formato corretto)
MODEL_URL = f"https://zulsyfmxuczxfkygphkb.supabase.co/storage/v1/object/public/models/{USER_ID}_model_sgd.pkl"
VECTORIZER_URL = f"https://zulsyfmxuczxfkygphkb.supabase.co/storage/v1/object/public/models/{USER_ID}_vectorizer_sgd.pkl"
