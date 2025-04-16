import os

# --- Supabase config ---
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://zulsyfmxuczxfkygphkb.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inp1bHN5Zm14dWN6eGZreWdwaGtiIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDA5MzkzMDgsImV4cCI6MjA1NjUxNTMwOH0.Mt_qQThNh2XZhaIu3Wezszg_63k7WuFFtzLAFMGSkJc")
SUPABASE_BUCKET = "ai-assets"

# --- Local paths ---
BASE_DIR = os.path.dirname(__file__)
TEMP_DIR = os.path.join(BASE_DIR, "temp_data")

MODEL_TEMPLATE = "{user_id}_model_sgd.pkl"
VECTORIZER_TEMPLATE = "{user_id}_vectorizer_sgd.pkl"
CORRECTIONS_TEMPLATE = "{user_id}_corrections.json"
STATS_TEMPLATE = "{user_id}_stats.txt"
ACCOUNTS_TEMPLATE = "{user_id}_accounts.json"

def normalize_user(user_id: str | None) -> str:
    return user_id or "default"

def get_model_path(user_id: str) -> str:
    return os.path.join(TEMP_DIR, MODEL_TEMPLATE.format(user_id=normalize_user(user_id)))

def get_vectorizer_path(user_id: str) -> str:
    return os.path.join(TEMP_DIR, VECTORIZER_TEMPLATE.format(user_id=normalize_user(user_id)))

def get_corrections_path(user_id: str) -> str:
    return os.path.join(TEMP_DIR, CORRECTIONS_TEMPLATE.format(user_id=normalize_user(user_id)))

def get_stats_path(user_id: str) -> str:
    return os.path.join(TEMP_DIR, STATS_TEMPLATE.format(user_id=normalize_user(user_id)))

def get_accounts_path(user_id: str) -> str:
    return os.path.join(TEMP_DIR, ACCOUNTS_TEMPLATE.format(user_id=normalize_user(user_id)))

def get_model_name(user_id: str) -> str:
    return MODEL_TEMPLATE.format(user_id=normalize_user(user_id))

def get_vectorizer_name(user_id: str) -> str:
    return VECTORIZER_TEMPLATE.format(user_id=normalize_user(user_id))

def get_corrections_name(user_id: str) -> str:
    return CORRECTIONS_TEMPLATE.format(user_id=normalize_user(user_id))

def get_stats_name(user_id: str) -> str:
    return STATS_TEMPLATE.format(user_id=normalize_user(user_id))

def get_accounts_name(user_id: str) -> str:
    return ACCOUNTS_TEMPLATE.format(user_id=normalize_user(user_id))
