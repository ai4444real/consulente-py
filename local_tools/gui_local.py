import tkinter as tk
from tkinter import filedialog, messagebox
import joblib
import pandas as pd
import numpy as np


current_model = None
current_vectorizer = None
current_description = ""

def load_model():
    """Apre una finestra per selezionare il file del modello"""
    global current_model
    model_path = filedialog.askopenfilename(filetypes=[("Pickle Files", "*.pkl")])
    if model_path:
        entry_model.delete(0, tk.END)
        entry_model.insert(0, model_path)
        current_model = joblib.load(model_path)

def load_vectorizer():
    """Apre una finestra per selezionare il file del vectorizer"""
    global current_vectorizer
    vectorizer_path = filedialog.askopenfilename(filetypes=[("Pickle Files", "*.pkl")])
    if vectorizer_path:
        entry_vectorizer.delete(0, tk.END)
        entry_vectorizer.insert(0, vectorizer_path)
        current_vectorizer = joblib.load(vectorizer_path)

def predict():
    """Esegue la predizione basata sul modello e il testo inserito"""
    global current_description
    description = entry_desc.get().strip()

    if not current_model or not current_vectorizer:
        messagebox.showerror("Errore", "Carica il modello e il vectorizer prima di procedere.")
        return

    if not description:
        messagebox.showwarning("Attenzione", "Inserisci una descrizione prima di fare la predizione.")
        return

    try:
        # Trasforma la descrizione in feature numeriche
        description_vectorized = current_vectorizer.transform([description])

        # Ora possiamo fare la predizione
        prediction = current_model.predict(description_vectorized)[0]

        # Mostra il risultato e lo mette nel campo di correzione
        label_result.config(text=f"✅ Conto Predetto: {prediction}", fg="green")
        entry_correction.delete(0, tk.END)
        entry_correction.insert(0, str(prediction))
        current_description = description

    except Exception as e:
        messagebox.showerror("Errore nella predizione", f"{e}")

def confirm_correction():
    """Registra la correzione manuale dell'utente"""
    corrected_value = entry_correction.get().strip()
    
    if not corrected_value:
        messagebox.showwarning("Attenzione", "Inserisci un valore per la correzione.")
        return

    label_correction.config(text=f"✅ Correzione registrata: {corrected_value}", fg="blue")

    # Aggiungere la correzione al modello (semplice aggiornamento dati in memoria)
    current_model.partial_fit(current_vectorizer.transform([current_description]), [corrected_value])

def test_correction():
    """Testa se il modello ha appreso la correzione"""
    if not current_description:
        messagebox.showwarning("Attenzione", "Fai prima una predizione per testare la correzione.")
        return

    try:
        description_vectorized = current_vectorizer.transform([current_description])
        new_prediction = current_model.predict(description_vectorized)[0]

        label_test_result.config(text=f"🔄 Nuova Predizione: {new_prediction}", fg="purple")

    except Exception as e:
        messagebox.showerror("Errore nel test", f"{e}")

def save_model():
    """Salva il modello aggiornato"""
    model_path = filedialog.asksaveasfilename(defaultextension=".pkl",
                                              filetypes=[("Pickle Files", "*.pkl")])
    if model_path:
        joblib.dump(current_model, model_path)
        messagebox.showinfo("Salvataggio", "Modello salvato con successo!")

def import_corrections():
    """Importa un file di corrections.json e aggiorna il modello"""
    global current_model, current_vectorizer

    if not current_model or not current_vectorizer:
        messagebox.showerror("Errore", "Carica prima il modello e il vectorizer.")
        return

    file_path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])

    if not file_path:
        return

    try:
        # Legge il file JSON delle corrections.json
        df = pd.read_json(file_path)

        # Controlla se ha le colonne corrette
        if not all(col in df.columns for col in ["Description", "Target_Account"]):
            messagebox.showerror("Errore", "Il file JSON deve contenere 'Description' e 'Target_Account'.")
            return

        # Vettorizza le descrizioni e aggiorna il modello
        X_train = current_vectorizer.transform(df["Description"])
        y_train = df["Target_Account"].astype(str)  # Assicura che le classi siano stringhe

        # Unisci tutte le classi conosciute con le nuove
        if hasattr(current_model, "classes_"):
            all_classes = np.unique(np.concatenate((current_model.classes_, np.unique(y_train))))
        else:
            all_classes = np.unique(y_train)

        # Verifica che tutte le classi di y_train siano presenti in all_classes
        missing_classes = set(y_train) - set(all_classes)
        if missing_classes:
            messagebox.showerror("Errore", f"Classi mancanti nel modello: {missing_classes}")
            return

        # Addestra il modello con tutte le classi possibili
        current_model.partial_fit(X_train, y_train, classes=all_classes)

        messagebox.showinfo("Successo", f"✅ {len(df)} corrections.json importate con successo!")
    
    except Exception as e:
        messagebox.showerror("Errore nell'importazione delle corrections.json", f"{e}")

from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

def create_new_model():
    """Crea un nuovo modello e vectorizer vuoti"""
    global current_model, current_vectorizer

    # Creiamo un modello SGDClassifier nuovo
    current_model = SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3)

    # Creiamo un vectorizer nuovo
    current_vectorizer = TfidfVectorizer()

    # Aggiorniamo la GUI per indicare che è stato creato un nuovo modello
    entry_model.delete(0, tk.END)
    entry_model.insert(0, "[Nuovo Modello]")

    entry_vectorizer.delete(0, tk.END)
    entry_vectorizer.insert(0, "[Nuovo Vectorizer]")

    messagebox.showinfo("Successo", "✅ Nuovo modello e vectorizer creati!")

def save_model_and_vectorizer():
    """Salva il modello e il vectorizer su file"""
    if not current_model or not current_vectorizer:
        messagebox.showerror("Errore", "Nessun modello da salvare. Crealo o caricalo prima.")
        return

    # Chiediamo dove salvare il modello
    model_path = filedialog.asksaveasfilename(defaultextension=".pkl",
                                              filetypes=[("Pickle Files", "*.pkl")],
                                              title="Salva Modello")

    if not model_path:
        return

    # Salviamo il modello
    joblib.dump(current_model, model_path)

    # Salviamo il vectorizer con lo stesso nome ma con _vectorizer
    vectorizer_path = model_path.replace(".pkl", "_vectorizer.pkl")
    joblib.dump(current_vectorizer, vectorizer_path)

    messagebox.showinfo("Successo", f"✅ Modello e vectorizer salvati:\n{model_path}\n{vectorizer_path}")

def train_model_from_file():
    """Carica un file JSON con corrections.json e allena il modello"""
    global current_model, current_vectorizer

    if not current_model or not current_vectorizer:
        messagebox.showerror("Errore", "Carica o crea un modello prima di allenarlo.")
        return

    file_path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])

    if not file_path:
        return

    try:
        # Carica il file JSON
        df = pd.read_json(file_path)

        # Controlla se ha le colonne corrette
        if not all(col in df.columns for col in ["Description", "Target_Account"]):
            messagebox.showerror("Errore", "Il file JSON deve contenere 'Description' e 'Target_Account'.")
            return

        # Vettorizza le descrizioni
        X_train = current_vectorizer.fit_transform(df["Description"])  # Fitta il vectorizer sui nuovi dati
        y_train = df["Target_Account"].astype(str)  # Converte le classi in stringhe per evitare errori

        # Se il modello è nuovo, inizializziamo le classi
        if not hasattr(current_model, "classes_"):
            all_classes = np.unique(y_train)
        else:
            all_classes = np.unique(np.concatenate((current_model.classes_, np.unique(y_train))))

        # Addestra il modello con tutte le classi possibili
        current_model.partial_fit(X_train, y_train, classes=all_classes)

        messagebox.showinfo("Successo", f"✅ Modello allenato con {len(df)} transazioni!")
    
    except Exception as e:
        messagebox.showerror("Errore", f"Errore nell'allenamento del modello:\n{e}")

def convert_csv_to_json():
    """Carica un file CSV e lo converte in JSON nel formato richiesto"""
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])

    if not file_path:
        return

    try:
        # Carica il file CSV
        separator = entry_separator.get().strip() or ";"  # Usa il valore inserito, default ";"

        df = pd.read_csv(file_path, delimiter=separator, encoding="ISO-8859-1", dtype={3: str}, header=None)

        # Assegna manualmente i nomi delle colonne
        df.columns = ["Date", "Description", "Importo", "Target_Account"]

        # Rinomina le colonne per il formato JSON
        df.rename(columns={
            "data": "Date",
            "descrizione": "Description",
            "importo": "Importo",
            "numero conto": "Target_Account"
        }, inplace=True)

        # Salva in JSON
        json_path = filedialog.asksaveasfilename(defaultextension=".json",
                                                 filetypes=[("JSON Files", "*.json")],
                                                 title="Salva il file JSON")

        if not json_path:
            return

        df.to_json(json_path, orient="records", indent=4, force_ascii=False)

        messagebox.showinfo("Successo", f"✅ JSON creato con successo!\nSalvato in: {json_path}")

    except Exception as e:
        messagebox.showerror("Errore nella conversione", f"{e}")


# Creazione finestra principale
root = tk.Tk()
root.title("Predizione Contabile")

# Selezione del modello
tk.Label(root, text="Modello:").grid(row=0, column=0, padx=5, pady=5)
entry_model = tk.Entry(root, width=40)
entry_model.grid(row=0, column=1, padx=5, pady=5)
tk.Button(root, text="Scegli", command=load_model).grid(row=0, column=2, padx=5, pady=5)

# Selezione del vectorizer
tk.Label(root, text="Vectorizer:").grid(row=1, column=0, padx=5, pady=5)
entry_vectorizer = tk.Entry(root, width=40)
entry_vectorizer.grid(row=1, column=1, padx=5, pady=5)
tk.Button(root, text="Scegli", command=load_vectorizer).grid(row=1, column=2, padx=5, pady=5)

# Inserimento descrizione
tk.Label(root, text="Descrizione:").grid(row=2, column=0, padx=5, pady=5)
entry_desc = tk.Entry(root, width=40)
entry_desc.grid(row=2, column=1, padx=5, pady=5)

# Bottone per la predizione
tk.Button(root, text="Predici", command=predict).grid(row=2, column=2, padx=5, pady=10)

# Label per il risultato della predizione
label_result = tk.Label(root, text="", fg="blue")
label_result.grid(row=3, column=1, padx=5, pady=5)

# Campo per correggere la predizione
tk.Label(root, text="Correggi la predizione:").grid(row=4, column=0, padx=5, pady=5)
entry_correction = tk.Entry(root, width=20)
entry_correction.grid(row=4, column=1, padx=5, pady=5)

# Bottone per confermare la correzione
tk.Button(root, text="Correggi", command=confirm_correction).grid(row=4, column=2, padx=5, pady=10)

# Bottone per testare la correzione
#tk.Button(root, text="Testa Nuovamente", command=test_correction).grid(row=7, column=1, padx=5, pady=10)

# Bottone per importare corrections.json
tk.Button(root, text="📂 Importa corrections.json", command=import_corrections).grid(row=6, column=1, padx=5, pady=10)

# Bottone per creare un nuovo modello
tk.Button(root, text="🆕 Crea Modello", command=create_new_model).grid(row=7, column=0, padx=5, pady=10)

# Bottone per allenare il modello con un file JSON
tk.Button(root, text="📂 Allena Modello", command=train_model_from_file).grid(row=7, column=1, padx=5, pady=10)

# Bottone per salvare il modello e vectorizer
tk.Button(root, text="💾 Salva Modello", command=save_model_and_vectorizer).grid(row=7, column=2, padx=5, pady=10)

# Bottone per convertire CSV in JSON
tk.Button(root, text="📂 CSV2JSON", command=convert_csv_to_json).grid(row=8, column=0, padx=5, pady=10)

# Campo per scegliere il separatore
tk.Label(root, text="Separatore CSV:").grid(row=8, column=1, padx=5, pady=5)
entry_separator = tk.Entry(root, width=3)
entry_separator.grid(row=8, column=2, padx=5, pady=5, sticky="w")
entry_separator.insert(0, ",")  # Imposta il valore predefinito


# Label per conferma della correzione
label_correction = tk.Label(root, text="", fg="blue")
label_correction.grid(row=7, column=1, padx=5, pady=5)

# Label per il risultato del test
label_test_result = tk.Label(root, text="", fg="purple")
label_test_result.grid(row=9, column=1, padx=5, pady=5)




# Avvia l'interfaccia grafica
root.mainloop()
