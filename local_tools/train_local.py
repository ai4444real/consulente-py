import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
import json
import os

# Chiede all'utente il file di training e il modello da aggiornare
training_file = input("📂 Inserisci il percorso del file di training (JSON): ").strip()
model_file = input("📂 Inserisci il percorso del modello da aggiornare (PKL): ").strip()

# Verifica che il file di training esista
if not os.path.exists(training_file):
    print(f"❌ Errore: Il file {training_file} non esiste.")
    exit(1)

# Carica il dataset di training
with open(training_file, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# Estrai X (descrizioni) e y (conti contabili)
X_train = [item["Description"] for item in dataset]
y_train = [item["Target_Account"] for item in dataset]

# Verifica che i dati siano coerenti
assert len(X_train) == len(y_train), "❌ Errore: X_train e y_train hanno dimensioni diverse"

# Se il modello esiste, caricalo e aggiorna il training
if os.path.exists(model_file):
    print(f"🔄 Aggiornamento del modello esistente: {model_file}")
    sgd_model = joblib.load(model_file)
    vectorizer = joblib.load("vectorizer_sgd.pkl")  # Assumiamo che il vectorizer sia sempre lo stesso

    # Trasforma i nuovi dati
    X_train_transformed = vectorizer.transform(X_train)

    # Continua l'addestramento con nuovi dati
    sgd_model.partial_fit(X_train_transformed, y_train)
else:
    print(f"🆕 Creazione di un nuovo modello.")
    vectorizer = TfidfVectorizer()
    X_train_transformed = vectorizer.fit_transform(X_train)

    # Crea un nuovo modello SGDClassifier
    sgd_model = SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3)

    # Creiamo una lista di classi completa con tutti i conti contabili
    class_labels = sorted(set(y_train))

    # Addestra il modello da zero
    sgd_model.partial_fit(X_train_transformed, y_train, classes=class_labels)

    # Salva anche il vettorizzatore per future previsioni
    joblib.dump(vectorizer, "vectorizer_sgd.pkl")

# Salva il modello aggiornato
joblib.dump(sgd_model, model_file)

print(f"✅ Modello aggiornato e salvato in '{model_file}'")
