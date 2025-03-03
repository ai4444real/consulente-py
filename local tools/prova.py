import pandas as pd

file_path = "export_all_transactions_2024.csv"

# Leggiamo il file senza modificare nulla, solo per vedere i nomi delle colonne
df = pd.read_csv(file_path, skiprows=6, delimiter=";", encoding="utf-8", quotechar='"')

# Stampiamo i nomi delle colonne per verificare
print("✅ Colonne rilevate nel file CSV:")
print(df.columns)
