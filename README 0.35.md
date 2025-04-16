# Consulente AI - Versione 0.35

Questo progetto fornisce una base modulare per servizi di Intelligenza Artificiale containerizzati, focalizzati su due funzionalità principali:

---

## ✅ Funzionalità principali

### 1. **Predizione contabile (servizio `predictor`)**
Dati in input (es. descrizione di una transazione bancaria), il sistema restituisce un codice conto suggerito.
- Input: `text` (es. "Pagamento Amazon")
- Output: `predictedAccount` (es. "6500")
- Allenabile su dataset differenti per altri compiti (intenzioni, categorie, ecc.)

### 2. **Lettura PDF (servizio `pdf-reader`)**
Estrae testo da PDF semplici o scansionati.
- Riconoscimento automatico: OCR solo se necessario
- Output: testo continuo

---

## 🐳 Deploy & Architettura

### Docker-ready
Tutti i servizi sono containerizzati e pronti per il deploy tramite:
- **Locale**
- **Fly.io** (o altra piattaforma compatibile Docker)

---

## 💻 Interfacce disponibili

### 1. `index.html` – Interfaccia utente
Permette di:
- Eseguire predizioni singole
- Inviare feedback
- Caricare file CSV per predizione batch
- Scaricare risorse (modello, vettorizzatore, correzioni)

> 🟢 Pronta per il deploy su frontend hosting (es. Vercel)

### 2. `testSuite.html` – Tester Curl
Test automatico degli endpoint tramite comandi `curl`, su ambienti:
- `localhost`
- `https://consulente-py.fly.dev`

---

## 📂 Struttura del progetto

```
predictor/           # servizio AI contabile (FastAPI)
pdf-reader/          # lettura PDF & OCR (FastAPI)
ui/                  # interfaccia HTML/JS
Dockerfile           # build multi-servizio
index.html           # interfaccia utente
testSuite.html       # tester automatico curl
config.js            # configurazione base URL
```

---

## 🔧 Requisiti per lo sviluppo locale

- Docker
- Python (solo per sviluppo, non necessario in produzione)
- Nessuna dipendenza di sistema extra (venv, pycache già esclusi)

---

## 📦 Versione attuale

**0.35 – Funziona tutto localmente e su Fly.io.**  
Predizione batch, test curl, UI interattiva, download dei file e PDF reader attivi.