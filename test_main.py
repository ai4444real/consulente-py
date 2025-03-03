import pytest
from fastapi.testclient import TestClient
from main import app

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
    print(response.json())  # Stampa il messaggio di errore

    assert response.status_code == 200
    assert response.json()["message"] == "Correzione registrata con successo!"


if __name__ == "__main__":
    pytest.main()
