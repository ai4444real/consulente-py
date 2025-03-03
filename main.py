from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import requests

accountDescriptions = {
    "1000": "Cassa (240.80.00)",
    "1001": "Cassa Euro (0.00)",
    "1020": "Conto Raiffeisen CHF - linea di credito",
    "1021": "Conto Raiffeisen EUR (-16.23 - 16.36 EUR)",
    "1023": "Conto Raiffeisen CHF (1925.46 - 1'446.47 CHF)",
    "1025": "Conto corrente postale (47'426.12 - 48'764.74 CHF)",
    "1026": "Conto corrente postale - riserve (3'365.25 - 15040.80 CHF)",
    "1027": "Paypal CHF (CHF 4.66)",
    "1090": "Conto di trasferimento",
    "1095": "cc Di Gregorio",
    "1100": "Clienti",
    "1109": "Delcredere",
    "1300": "Costi anticipati",
    "1301": "Ricavi da incassare",
    "1510": "Mobilio e installazioni",
    "1520": "Macchine ufficio, informatica e comunicazione",
    "1526": "Software",
    "2000": "Creditori",
    "2050": "cc AVS",
    "2051": "cc Cassa Pensione LPP",
    "2052": "cc malattia",
    "2053": "cc Zurich infortuni",
    "2054": "cc Imposte alla fonte",
    "2055": "cc Indennità maternità",
    "2056": "cc Cassa assegni familiari",
    "2062": "cc Stipendi altri",
    "2070": "cc Carta di credito",
    "2099": "cc Servizio di fatturazione",
    "2261": "Dividendi",
    "2281": "Imposta preventiva su dividendi",
    "2600": "Accantonamenti",
    "2730": "Transitori passivi (non ancora pagati)",
    "2800": "Capitale proprio / capitale sociale",
    "2900": "Riserva legale da capitale",
    "2990": "Utile o perdita riportata",
    "3400": "Ricavi da corsi di formazione",
    "3401": "Ricavi da servizio di fatturazione",
    "3805": "Perdite su debitori",
    "6850": "Interessi Attivi",
    "8510": "Ricavi straordinari",
    "4400": "Costi per prestazioni di terzi",
    "4401": "Costi per consulenti interni",
    "4402": "Costi ausiliari per prestazioni di terzi",
    "4430": "Dispense",
    "4460": "Costi associativi",
    "5000": "Salari",
    "5001": "Premi prestazioni",
    "5700": "Contributi AVS/AI/IPG",
    "5720": "Contributi previdenza professionale",
    "5730": "Assicurazione infortuni",
    "5740": "Premi assicurazione malattia",
    "5750": "Indennità maternità",
    "5760": "Indennità COVID19",
    "5790": "Imposta alla fonte",
    "5820": "Spese di viaggio",
    "5830": "Rimborsi spese forfettarie",
    "5870": "Costi assicurativi",
    "5880": "Altre spese personale (costi senza cliente)",
    "5890": "Costi formazione ed aggiornamento",
    "6000": "Pigione/affitto aule",
    "6105": "Costi leasing immobilizzazioni mobiliari",
    "6400": "Costi energia",
    "6500": "Spese ufficio/amministrazione",
    "6550": "Spese di consulenza amministrativa",
    "6551": "Spese telefoniche/internet",
    "6552": "Spese postali",
    "6600": "Costi di pubblicità e marketing",
    "6650": "Spese di rappresentanza",
    "6660": "Costi software",
    "6700": "Altri costi",
    "6800": "Interessi e spese bancarie",
    "6842": "Differenze di cambio/Arrotondamenti",
    "6900": "Ammortamenti",
    "8500": "Costi straordinari",
    "8900": "Imposte"
}

# URL dei file su Supabase (SOSTITUISCI con gli URL reali)
SGD_MODEL_URL = "https://zulsyfmxuczxfkygphkb.supabase.co/storage/v1/object/public/models//modello_sgd.pkl"
VECTORIZER_URL = "https://zulsyfmxuczxfkygphkb.supabase.co/storage/v1/object/public/models//vectorizer_sgd.pkl"

# Scarica i file da Supabase
def download_file(url, filename):
    response = requests.get(url)
    with open(filename, "wb") as f:
        f.write(response.content)

download_file(SGD_MODEL_URL, "modello_sgd.pkl")
download_file(VECTORIZER_URL, "vectorizer_sgd.pkl")

# Carichiamo il modello e il vectorizer
sgd_model = joblib.load("modello_sgd.pkl")
vectorizer_sgd = joblib.load("vectorizer_sgd.pkl")

print("✅ Modello SGDClassifier caricato con successo!")

# Inizializziamo FastAPI
app = FastAPI()

# Abilitiamo i CORS per consentire richieste dall'interfaccia web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Puoi restringere ai tuoi domini se necessario
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Definizione della struttura dei dati in ingresso
class Transaction(BaseModel):
    description: str
    amount: float
    correctAccount: str = None 

# Endpoint per predire il conto contabile
@app.post("/predict")
def predict(transaction: Transaction):
    try:
        # Convertiamo la descrizione in numeri con il vectorizer
        features = vectorizer_sgd.transform([transaction.description])

        # Prediciamo il conto contabile
        predicted_account = sgd_model.predict(features)[0]

        return {
            "description": transaction.description,
            "amount": transaction.amount,
            "predictedAccount": predicted_account
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


import json
CLASS_FILE = "classes.json"  # File per memorizzare le classi viste dal modello

# Funzione per caricare le classi dal file
def load_classes():
    try:
        with open(CLASS_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return list(sgd_model.classes_)  # Se il file non esiste, usiamo le classi attuali

# Funzione per salvare le classi aggiornate
def save_classes(classes):
    with open(CLASS_FILE, "w") as f:
        json.dump(classes, f)

@app.post("/feedback")
def feedback(transaction: Transaction):
    try:
        # Convertiamo la descrizione in una rappresentazione numerica
        features = vectorizer_sgd.transform([transaction.description])

        # Controlliamo se il conto corretto è già nel modello
        current_classes = set(map(str, sgd_model.classes_))
        print(f"⚠️ Classi attuali nel modello: {current_classes}")
        print(f"✅ Nuova classe da aggiungere: {transaction.correctAccount}")

        # Se la nuova classe è già conosciuta, aggiorniamo senza toccare le classi
        if str(transaction.correctAccount) in current_classes:
            print("🔄 Aggiornamento senza modificare le classi esistenti.")
            sgd_model.partial_fit(features, [transaction.correctAccount])  # Senza specificare le classi
        else:
            # Caso raro: se una classe nuova viene aggiunta, dobbiamo riallenare con TUTTE le classi
            print("⚠️ Nuova classe rilevata! Riaddestriamo il modello con tutte le classi conosciute.")
            new_classes = sorted(current_classes | {str(transaction.correctAccount)})
            sgd_model.partial_fit(features, [transaction.correctAccount], classes=new_classes)

        # Salviamo il modello aggiornato
        joblib.dump(sgd_model, "modello_sgd.pkl")

        return {"message": "Correzione registrata con successo!"}
    except Exception as e:
        print(f"❌ Errore durante l'aggiornamento: {e}")
        raise HTTPException(status_code=500, detail=str(e))



# Endpoint di test
@app.get("/")
def home():
    return {"message": "API di Predizione Contabile attiva con SGDClassifier"}
