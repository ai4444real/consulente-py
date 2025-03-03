import joblib
import os
import numpy as np

# Percorsi dei file
MODEL_PATH = "modello_sgd.pkl"
VECTORIZER_PATH = "vectorizer_sgd.pkl"

# Caricare modello e vettorizzatore una volta sola
if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    print("❌ Errore: Modello o vettorizzatore non trovati. Assicurati di averli scaricati!")
    exit(1)

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

def predici_conto(description):
    """
    Predice il conto contabile per una nuova transazione.
    """
    try:
        # Vettorizzare la descrizione
        features = vectorizer.transform([description])

        # Fare la previsione
        prediction = model.predict(features)[0]

        print(f"✅ Conto contabile suggerito: {prediction}")
        return prediction
    except Exception as e:
        print(f"❌ Errore durante la predizione: {e}")
        return None

# Esempio di utilizzo
if __name__ == "__main__":
    descrizione = input("Inserisci la descrizione della transazione: ")
    predici_conto(descrizione)
