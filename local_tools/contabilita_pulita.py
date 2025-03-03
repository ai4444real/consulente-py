import pandas as pd

def load_contabilita_csv(file_path):
    """
    Carica e pulisce il file CSV delle registrazioni contabili.
    """
    try:
        df = pd.read_csv(file_path, delimiter=",", encoding="utf-8", quotechar='"')
        
        # Convertire le date in formato standard YYYY-MM-DD
        df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d").dt.strftime("%Y-%m-%d")
        
        # Rimuovere eventuali apostrofi nei numeri e convertire in float
        df["Credit"] = pd.to_numeric(df["Credit"].astype(str).str.replace("'", ""), errors="coerce").fillna(0.00)
        df["Debit"] = pd.to_numeric(df["Debit"].astype(str).str.replace("'", ""), errors="coerce").fillna(0.00)
        
        # Invertire Credit e Debit per riflettere la realtà contabile corretta
        df["Credit"], df["Debit"] = df["Debit"], df["Credit"]
        
        # Assegna il segno corretto agli importi: Crediti positivi, Debiti negativi
        df["Importo"] = df["Credit"]
        df.loc[df["Debit"] > 0, "Importo"] = -df["Debit"]
        
        # DEBUG: Verifica importi
        print("✅ Controllo Importi (primi 20 valori)")
        print(df[["Debit", "Credit", "Importo"]].head(20))
        
        # Mantenere solo le colonne essenziali
        df_cleaned = df[["Date", "Description", "Importo", "AccountSelected", "ContraAccount"]]
        
        # Salvare il file pulito
        df_cleaned.to_csv("contabilita_pulita.csv", index=False, encoding="utf-8")
        
        print("✅ File contabile pulito salvato come 'contabilita_pulita.csv'")
        return df_cleaned
    except Exception as e:
        print(f"Errore durante la lettura del CSV contabile: {e}")
        return None

# Nome del file CSV con i dati contabili
contabilita_file_path = "postfinance 2024.csv"

# Caricare e pulire il file delle registrazioni contabili
df_contabilita = load_contabilita_csv(contabilita_file_path)
