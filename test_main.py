import pytest
import json
import os
from fastapi.testclient import TestClient
from main import app, CORRECTIONS_FILE

client = TestClient(app)

# Test per verificare che l'API sia attiva
def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API di Predizione Contabile attiva con SGDClassifier"}

# Test per verificare la predizione
def test_predict():
    sample_transaction = {
        "description": "Acquisto cancelleria",
        "amount": 45.30
    }
    response = client.post("/predict", json=sample_transaction)
    assert response.status_code == 200
    assert "predictedAccount" in response.json()

# Test per il feedback (simulazione di aggiornamento del modello)
def test_feedback():
    sample_feedback = {
        "description": "Acquisto hardware",
        "amount": 199.99,
        "correctAccount": "6650"
    }
    response = client.post("/feedback", json=sample_feedback)
    assert response.status_code == 200
    assert response.json()["message"] == "Correzione registrata con successo!"

# Test per verificare che la correzione venga salvata in correzioni.json
def test_correction_saved():
    sample_feedback = {
        "description": "Test correzione",
        "amount": 10.50,
        "correctAccount": "5700"
    }

    # Assicuriamoci che il file non esista prima del test
    if os.path.exists(CORRECTIONS_FILE):
        os.remove(CORRECTIONS_FILE)

    # Inviamo il feedback
    response = client.post("/feedback", json=sample_feedback)
    assert response.status_code == 200

    # Controlliamo che il file sia stato creato
    assert os.path.exists(CORRECTIONS_FILE), "❌ Il file correzioni.json non è stato creato"

    # Controlliamo che il contenuto sia corretto
    with open(CORRECTIONS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) > 0, "❌ Il file correzioni.json è vuoto"
    assert data[-1]["Description"] == "Test correzione"
    assert data[-1]["Importo"] == 10.50
    assert data[-1]["Target_Account"] == "5700"

if __name__ == "__main__":
    pytest.main()
