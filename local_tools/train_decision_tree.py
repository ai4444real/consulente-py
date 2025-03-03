import pandas as pd
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

# Caricare il dataset
with open("dataset_ai.json", "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Features (X) e Target (y)
X = df[["Description", "Importo"]]
y = df["Target_Account"]

# Suddivisione training e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline: Trasforma il testo e allena un Decision Tree
model = make_pipeline(
    TfidfVectorizer(),  # Trasforma il testo in vettori numerici
    DecisionTreeClassifier(random_state=42)  # Modello base
)

# Addestramento del modello
model.fit(X_train["Description"], y_train)

# Test del modello
predictions = model.predict(X_test["Description"])
accuracy = accuracy_score(y_test, predictions)

# Salvataggio del modello addestrato
joblib.dump(model, "modello_ai.pkl")

print(f"✅ Modello addestrato e salvato! Precisione: {accuracy:.2%}")



