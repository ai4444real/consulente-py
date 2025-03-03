import pandas as pd

def load_bank_csv(file_path):
    """
    Carica un file CSV di estratto bancario, ignorando intestazione e footer.
    """
    try:
        # Legge il CSV ignorando le prime 6 righe
        df = pd.read_csv(file_path, skiprows=6, delimiter=";", encoding="utf-8", quotechar='"')

        # Rimuove righe completamente vuote
        df = df.dropna(how="all")

        # Assegna nomi corretti alle colonne
        df.columns = ["Date", "Notification text", "Credit in CHF", "Debit in CHF", "Value", "Balance in CHF"]

        # Rimuove la colonna Balance in CHF (non necessaria)
        df = df.drop(columns=["Balance in CHF"])

        return df
    except Exception as e:
        print(f"Errore durante la lettura del CSV: {e}")
        return None

# Nome del file CSV (deve essere nella stessa cartella dello script)
file_path = "export_all_transactions_2024.csv"

df = load_bank_csv(file_path)

if df is not None:
    # Mantiene solo le righe in cui la colonna "Date" è effettivamente una data
    df = df[df["Date"].str.match(r"\d{2}\.\d{2}\.\d{4}") == True]

    # Converte il formato delle date
    df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y").dt.strftime("%Y-%m-%d")

    # Converte gli importi in numeri (forza i NaN a 0.00)
    df["Credit in CHF"] = pd.to_numeric(df["Credit in CHF"], errors="coerce").fillna(0.00)
    df["Debit in CHF"] = pd.to_numeric(df["Debit in CHF"], errors="coerce").fillna(0.00)

    # Crea una colonna "Importo" unica, mantenendo il segno corretto
    df["Importo"] = df["Credit in CHF"] - df["Debit in CHF"].abs()

    # DEBUG: Verifica importi
    print("✅ Controllo Importi (primi 20 valori)")
    print(df[["Credit in CHF", "Debit in CHF", "Importo"]].head(20))

    # Mantiene solo le colonne essenziali
    df_cleaned = df[["Date", "Notification text", "Importo"]]

    # Salva il file pulito come CSV
    df_cleaned.to_csv("transazioni_pulite.csv", index=False, encoding="utf-8")

    print("✅ File pulito salvato come 'transazioni_pulite.csv'")
