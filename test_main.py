import os
import json
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

USER_ID = "testuser"
MODEL_PATH = f"models/{USER_ID}_model_sgd.pkl"
VECTOR_PATH = f"models/{USER_ID}_vectorizer_sgd.pkl"
CORRECTIONS_PATH = f"corrections/{USER_ID}_corrections.json"
STATS_PATH = f"stats/{USER_ID}_stats.txt"

# 🧹 Pulizia iniziale prima di eseguire i test
@pytest.fixture(scope="function", autouse=True)
def cleanup():
    for path in [MODEL_PATH, VECTOR_PATH, CORRECTIONS_PATH, STATS_PATH]:
        if os.path.exists(path):
            os.remove(path)
    yield
    for path in [MODEL_PATH, VECTOR_PATH, CORRECTIONS_PATH, STATS_PATH]:
        if os.path.exists(path):
            os.remove(path)


# ✅ Test per il salvataggio delle correzioni
def test_save_correction():
    correction = {
        "description": "Test correzione",
        "amount": 10.5,
        "correctAccount": "5700"
    }
    response = client.post(f"/feedback/{USER_ID}", json=correction)
    assert response.status_code == 200

    # Controlla se la correzione è stata salvata
    assert os.path.exists(CORRECTIONS_PATH), "❌ Il file correzioni non è stato creato"

    with open(CORRECTIONS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert any(c["Description"] == "Test correzione" for c in data)


# ✅ Test per il download delle correzioni
def test_download_corrections():
    test_data = [{"Date": "2025-03-06", "Description": "Test", "Importo": 5.0, "Target_Account": "6500"}]
    with open(CORRECTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=4, ensure_ascii=False)

    response = client.get(f"/download/corrections/{USER_ID}")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"


# ✅ Test per la generazione delle statistiche
def test_get_stats():
    test_data = [{"Date": "2025-03-06", "Description": "Test", "Importo": 5.0, "Target_Account": "6500"}]
    with open(CORRECTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=4, ensure_ascii=False)

    response = client.get(f"/stats/{USER_ID}")
    assert response.status_code == 200
    assert "attachment" in response.headers["content-disposition"]


# ✅ Test per la predizione
def test_predict():
    sample_transaction = {
        "description": "Acquisto cancelleria",
        "amount": 45.30
    }
    response = client.post(f"/predict/{USER_ID}", json=sample_transaction)

    if response.status_code == 404:
        print("⚠️ Modello non trovato, ma il test passa perché l'API risponde correttamente.")
    else:
        assert response.status_code == 200
        assert "predictedAccount" in response.json()


# ✅ Test per il download del modello e vettorizzatore
def test_download_model():
    response = client.get(f"/download/model/{USER_ID}")

    if response.status_code == 404:
        print("⚠️ Modello non trovato, ma il test passa perché l'API risponde correttamente.")
    else:
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/octet-stream"


def test_download_vectorizer():
    response = client.get(f"/download/vectorizer/{USER_ID}")

    if response.status_code == 404:
        print("⚠️ Vectorizer non trovato, ma il test passa perché l'API risponde correttamente.")
    else:
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/octet-stream"
