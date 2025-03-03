import joblib
import pandas as pd

def predici_conto(description, importo):
    """
    Carica il modello e predice il conto contabile per una nuova transazione.
    """
    try:
        # Caricare il modello salvato
        model = joblib.load("modello_ai.pkl")
        
        # Creare un DataFrame con la nuova transazione
        df_nuovo = pd.DataFrame({"Description": [description], "Importo": [importo]})
        
        # Fare la previsione
        prediction = model.predict(df_nuovo["Description"])[0]
        
        print(f"✅ Conto contabile suggerito: {prediction}")
        return prediction
    except Exception as e:
        print(f"Errore durante la predizione: {e}")
        return None

# Esempio di utilizzo
if __name__ == "__main__":
    descrizione = input("Inserisci la descrizione della transazione: ")
    importo = float(input("Inserisci l'importo della transazione: "))
    
    predici_conto(descrizione, importo)
