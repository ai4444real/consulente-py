import joblib
import pandas as pd
import numpy as np

def predici_conti_da_file(file_banca, output_file):
    """
    Carica il modello e predice i conti contabili per un file di transazioni bancarie.
    """
    try:
        # Caricare il modello salvato
        model = joblib.load("modello_ai.pkl")
        
        # Caricare il file delle transazioni bancarie ignorando le prime 6 righe
        df_banca = pd.read_csv(file_banca, skiprows=6, delimiter=";", encoding="utf-8", quotechar='"')
        
        # Rimuovere righe completamente vuote
        df_banca = df_banca.dropna(how="all")
        
        # Assegnare nomi corretti alle colonne
        df_banca.columns = ["Date", "Notification text", "Credit in CHF", "Debit in CHF", "Value", "Balance in CHF"]
        
        # Rimuovere la colonna Balance in CHF (non necessaria)
        df_banca = df_banca.drop(columns=["Balance in CHF"])
        
        # Assicurarsi che la colonna 'Notification text' esista
        if "Notification text" not in df_banca.columns:
            raise ValueError("La colonna 'Notification text' non è presente nel file bancario.")
        
        # Sostituire i valori NaN con stringa vuota per evitare errori (correzione per Pandas 3.0)
        df_banca = df_banca.copy()
        df_banca.loc[:, "Notification text"] = df_banca["Notification text"].fillna("")
        
        # Predire i conti contabili
        df_banca["Conto_Predetto"] = model.predict(df_banca["Notification text"])
        
        # Salvare il risultato
        df_banca.to_csv(output_file, index=False, encoding="utf-8")
        
        print(f"✅ File con conti predetti salvato come '{output_file}'")
        return df_banca
    except Exception as e:
        print(f"Errore durante la predizione batch: {e}")
        return None

# Esempio di utilizzo
if __name__ == "__main__":
    file_banca = "export_all_transactions_2024.csv"  # File originale dell'estratto conto
    output_file = "transazioni_predette.csv"
    
    predici_conti_da_file(file_banca, output_file)
