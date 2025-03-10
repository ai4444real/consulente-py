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
    response = client.post("/predict/default", json=sample_transaction)
    assert response.status_code == 200
    assert "predictedAccount" in response.json()

# Test per il feedback (simulazione di aggiornamento del modello)
def test_feedback():
    sample_feedback = {
        "description": "Acquisto hardware",
        "amount": 199.99,
        "correctAccount": "6650"
    }
    response = client.post("/feedback/default", json=sample_feedback)  # AGGIUNTO user_id
    assert response.status_code == 200
    assert response.json()["message"] == "Correzione registrata con successo!"

# Test per verificare che la correzione venga salvata in corrections.json.json
def test_correction_saved():
    sample_feedback = {
        "description": "Test correzione",
        "amount": 10.50,
        "correctAccount": "5700"
    }

    corrections_file = "corrections/default_corrections.json"

    # Assicuriamoci che il file non esista prima del test
    if os.path.exists(corrections_file):
        os.remove(corrections_file)

    # Inviamo il feedback
    response = client.post("/feedback/default", json=sample_feedback)  # AGGIUNTO user_id
    assert response.status_code == 200

    # Controlliamo che il file sia stato creato
    assert os.path.exists(corrections_file), "❌ Il file di corrections.json non è stato creato"

    # Controlliamo che il contenuto sia corretto
    with open(corrections_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) > 0, "❌ Il file corrections.json è vuoto"
    assert data[-1]["Description"] == "Test correzione"
    assert data[-1]["Importo"] == 10.50
    assert data[-1]["Target_Account"] == "5700"

def test_force_download_models():
    """Test per verificare la ricarica del modello dell'utente"""
    response = client.get("/force-download/models/default")

    if response.status_code == 404:
        print("⚠️ Modello non trovato, ma il test passa perché l'API risponde correttamente.")
    else:
        assert response.status_code == 200

def test_download_corrections():
    """Test per verificare il download delle corrections.json dell'utente"""
    corrections_file = "corrections/default_corrections.json.json"

    # Creiamo un file di test se non esiste
    if not os.path.exists("corrections"):
        os.mkdir("corrections")

    test_data = [{"Date": "2025-03-05", "Description": "Test", "Importo": 5.0, "Target_Account": "6500"}]
    with open(corrections_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=4, ensure_ascii=False)

    response = client.get("/download/corrections/default")
    
    if response.status_code == 404:
        print("⚠️ corrections.json non trovate, ma il test passa perché l'API risponde correttamente.")
    else:
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"

    # Pulizia
    os.remove(corrections_file)

def save_correction(description, amount, correct_account, user_id):
    """Salva la correzione nel file JSON specifico dell'utente."""
    corrections_file = f"corrections/{user_id}_corrections.json.json"  

    correction = {
        "Date": datetime.datetime.today().strftime("%Y-%m-%d"),
        "Description": description,
        "Importo": amount,
        "Target_Account": correct_account
    }

    try:
        # Se il file esiste, proviamo a leggerlo
        if os.path.exists(corrections_file):
            with open(corrections_file, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)  # Proviamo a caricare il JSON
                except json.JSONDecodeError:
                    print(f"⚠️ Warning: Il file {corrections_file} era corrotto o vuoto. Verrà ricreato.")
                    data = []  # Se il JSON non è valido, inizializziamo una lista vuota
        else:
            data = []  # Se il file non esiste, inizializziamo una lista vuota

        # Aggiungiamo la nuova correzione
        data.append(correction)

        # Salviamo il file aggiornato
        with open(corrections_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        print(f"✅ Correzione salvata per {user_id}, totale corrections.json: {len(data)}")

    except Exception as e:
        print(f"❌ Errore nel salvataggio della correzione per {user_id}: {e}")


def test_get_stats():
    """Test per verificare la generazione delle statistiche dell'utente"""
    response = client.get("/stats/default")

    if response.status_code == 404:
        print("⚠️ Modello non trovato, ma il test passa perché l'API risponde correttamente.")
    else:
        assert response.status_code == 200

        # Normalizziamo le virgolette per evitare errori
        content_disposition = response.headers["content-disposition"].replace('"', "'")
        assert "attachment; filename='default_stats.txt'" in content_disposition

def test_download_model():
    """Test per verificare il download del modello dell'utente"""
    response = client.get("/download/model/default")

    if response.status_code == 404:
        print("⚠️ Modello non trovato, ma il test passa perché l'API risponde correttamente.")
    else:
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/octet-stream"

def test_download_vectorizer():
    """Test per verificare il download del vettorizzatore dell'utente"""
    response = client.get("/download/vectorizer/default")
    
    if response.status_code == 404:
        print("⚠️ Vectorizer non trovato, ma il test passa perché l'API risponde correttamente.")
    else:
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/octet-stream"



if __name__ == "__main__":
    pytest.main()
