import os
from predictor.config import (
    SUPABASE_URL,
    SUPABASE_KEY,
    SUPABASE_BUCKET,
    TEMP_DIR,
    get_model_path, get_model_name,
    get_vectorizer_path, get_vectorizer_name,
    get_corrections_path, get_corrections_name,
    get_stats_path, get_stats_name,
    get_accounts_path, get_accounts_name,
    normalize_user
)

print("[DEBUG] storage.py importato correttamente")

try:
    from supabase import create_client
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    print(f"[ERROR] Supabase client init failed: {e}")


def download_file_from_supabase(user_id, file_name, destination_path):
    from predictor.config import SUPABASE_URL, SUPABASE_KEY, SUPABASE_BUCKET
    import requests

    url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{file_name}"
    headers = {"apikey": SUPABASE_KEY}
    print(f"[📁] Tentativo download di '{file_name}' in '{destination_path}'")

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        with open(destination_path, "wb") as f:
            f.write(response.content)
        print(f"[✓] File salvato: {destination_path}")
    else:
        print(f"[‼️] Errore durante il download di '{file_name}': {response.json()}")

def lazy_download_file(user_id, filename, destination_path):
    if not os.path.exists(destination_path):
        print(f"📥 [Lazy Load] {filename} non trovato localmente, scarico da Supabase...")
        download_file_from_supabase(user_id, filename, destination_path)
    return destination_path

def download_all_user_files(user_id: str):
    user_id = normalize_user(user_id)
    print(f"[🚀] Avvio download per l'utente: {user_id}")

    download_file_from_supabase(user_id, get_model_path(user_id), get_model_name(user_id))
    download_file_from_supabase(user_id, get_vectorizer_path(user_id), get_vectorizer_name(user_id))
    download_file_from_supabase(user_id, get_corrections_path(user_id), get_corrections_name(user_id))
    download_file_from_supabase(user_id, get_stats_path(user_id), get_stats_name(user_id))
    download_file_from_supabase(user_id, get_accounts_path(user_id), get_accounts_name(user_id))
