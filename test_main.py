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

def test_force_download_models():
    """Test per forzare il download e reload del modello"""
    response = client.get("/force-download/models")
    assert response.status_code == 200

def test_download_corrections():
    """Test per scaricare il file delle correzioni"""
    # Creiamo un file di test
    test_data = [{"Date": "2025-03-05", "Description": "Test", "Importo": 5.0, "Target_Account": "6500"}]
    with open(CORRECTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=4, ensure_ascii=False)

    response = client.get("/download/corrections")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"

    # Pulizia
    os.remove(CORRECTIONS_FILE)

def test_save_correction():
    """Test per verificare il salvataggio di una correzione"""
    correction = {
        "description": "Test correzione",
        "amount": 10.5,
        "correctAccount": "5700"
    }
    response = client.post("/feedback", json=correction)
    assert response.status_code == 200

    # Controlla se la correzione è stata salvata
    with open(CORRECTIONS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert any(c["Description"] == "Test correzione" for c in data)

    # Pulizia
    os.remove(CORRECTIONS_FILE)

def test_get_stats():
    """Test per verificare la generazione delle statistiche"""
    response = client.get("/stats")
    assert response.status_code == 200
    assert "attachment; filename=server_stats.txt" in response.headers["content-disposition"]






if __name__ == "__main__":
    pytest.main()
