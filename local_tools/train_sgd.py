import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
import json

# Carica il file JSON
with open("../accounts.json", "r", encoding="utf-8") as file:
    accountDescriptions = json.load(file)

# Carichiamo il dataset di training (assumiamo che sia già in formato JSON)
with open("dataset_ai.json", "r") as f:
    dataset = json.load(f)

# Estriamo X (descrizioni) e y (conti contabili)
X_train = [item["Description"] for item in dataset]
y_train = [item["Target_Account"] for item in dataset]

# Verifichiamo che i dati siano coerenti
assert len(X_train) == len(y_train), "Errore: X_train e y_train hanno dimensioni diverse"

# Convertiamo le descrizioni in numeri con TfidfVectorizer
vectorizer = TfidfVectorizer()
X_train_transformed = vectorizer.fit_transform(X_train)

# Creiamo e alleniamo il modello SGDClassifier
sgd_model = SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3)

# Creiamo una lista di classi completa con tutti i conti contabili
class_labels = sorted(set(y_train) | set(accountDescriptions.keys()))

# Addestriamo il modello con tutte le classi, anche se alcune non compaiono nei dati
sgd_model.partial_fit(X_train_transformed, y_train, classes=class_labels)


# Salviamo il modello e il vectorizer
joblib.dump(sgd_model, "modello_sgd.pkl")
joblib.dump(vectorizer, "vectorizer_sgd.pkl")

print("✅ Modello SGDClassifier salvato come modello_sgd.pkl")
