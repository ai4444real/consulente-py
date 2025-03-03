import pandas as pd
import json

def confronta_transazioni(file_banca, file_contabilita, output_json):
    """
    Confronta le transazioni tra estratto conto bancario e contabilità,
    ignorando transazioni isolate (non abbinate) e preparando il dataset per l'IA.
    """
    try:
        df_banca = pd.read_csv(file_banca, encoding="utf-8")
        df_contabilita = pd.read_csv(file_contabilita, encoding="utf-8")
        
        # Unire i due dataset per identificare differenze
        df_confronto = df_banca.merge(df_contabilita, on=["Date", "Importo"], how="inner")
        
        # Creazione dataset per IA con feature selezionate
        df_ai = df_confronto.loc[:, ["Date", "Notification text", "Importo", "ContraAccount"]].copy()
        df_ai.rename(columns={"Notification text": "Description", "ContraAccount": "Target_Account"}, inplace=True)
        
        # Salvare il file JSON per il training AI
        df_ai.to_json(output_json, orient="records", indent=4, force_ascii=False)
        print(f"✅ Dataset AI salvato come '{output_json}'")
        
        return df_ai
    except Exception as e:
        print(f"Errore durante il confronto: {e}")
        return None

# Nome dei file CSV
file_banca = "transazioni_pulite.csv"
file_contabilita = "contabilita_pulita.csv"
output_json = "dataset_ai.json"

# Confrontare i due file e generare dataset AI
df_ai = confronta_transazioni(file_banca, file_contabilita, output_json)
