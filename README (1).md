# 🛡️ OXVENOM — AI-Based Network Intrusion Detection System

> A machine learning powered Network Intrusion Detection System (NIDS) that classifies network traffic as **Normal** or **Attack** in real-time using models trained on the NSL-KDD dataset.

---

## 📌 Project Overview

OXVENOM is a full-stack intrusion detection system built as part of a college project. It combines classical machine learning with a modern web dashboard to monitor, detect, and alert on suspicious network activity.

The system trains multiple ML models on the NSL-KDD dataset, selects the best performing model, and exposes it via a REST API that powers a real-time monitoring dashboard.

---

## 🚀 Features

- ✅ **Real-time Monitoring** — Simulates network traffic and classifies each packet as Normal or Attack
- ✅ **Multiple ML Models** — Trains and compares Random Forest, Gradient Boosting, Logistic Regression, and KNN
- ✅ **Live Training Logs** — Watch the model train in real-time from the dashboard
- ✅ **REST API** — Flask-powered API with API key authentication
- ✅ **Interactive Dashboard** — Charts, alerts, analytics, and system status in one place
- ✅ **File Upload** — Upload custom training/testing datasets via the dashboard
- ✅ **Analytics Page** — Model performance comparison, attack distribution, accuracy over time

---

## 🧠 ML Models & Results

All models trained on the **NSL-KDD dataset** (125,973 training records, 41 features).

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| **Random Forest** ⭐ | 99.90% | 99.94% | 99.85% | 99.89% |
| Gradient Boosting | 99.66% | 99.73% | 99.54% | 99.63% |
| KNN | 99.60% | 99.62% | 99.52% | 99.57% |
| Logistic Regression | 95.21% | 95.86% | 93.75% | 94.79% |

**Best Model: Random Forest with 99.9% accuracy**

---

## 🗂️ Project Structure

```
AI_IDS_PROJECT/
│
├── api/
│   └── app.py              # Flask REST API (port 5000)
│
├── dashboard/
│   └── app.py              # Dashboard server (port 8080)
│
├── src/
│   ├── predict.py          # Prediction module
│   ├── preprocess.py       # Data preprocessing pipeline
│   └── train_model.py      # Model training module
│
├── models/
│   ├── best_model.pkl      # Saved best model (Random Forest)
│   ├── preprocessor.pkl    # Saved preprocessor
│   └── results.json        # Training results
│
├── data/
│   └── raw/
│       ├── KDDTrain+.txt   # NSL-KDD training dataset
│       └── KDDTest+.txt    # NSL-KDD testing dataset
│
├── index.html              # Main dashboard UI
└── README.md               # This file
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| ML Models | Scikit-learn (Random Forest, Gradient Boosting, Logistic Regression, KNN) |
| Backend API | Python, Flask, Flask-CORS |
| Frontend | HTML, CSS, JavaScript, Chart.js |
| Data Processing | Pandas, NumPy |
| Model Storage | Joblib |
| Dataset | NSL-KDD (Network Security Lab Knowledge Discovery Dataset) |

---

## 🔧 How to Run

### Prerequisites
- Python 3.x
- Virtual environment (venv)
- NSL-KDD dataset in `data/raw/`

### Step 1 — Clone / Download the project
```
cd C:\path\to\AI_IDS_PROJECT
```

### Step 2 — Install dependencies
```
.\venv\Scripts\pip.exe install -r requirements.txt
```

### Step 3 — Start the API server (Terminal 1)
```
.\venv\Scripts\python.exe api\app.py
```
API will be running at: `http://127.0.0.1:5000`

### Step 4 — Start the Dashboard server (Terminal 2)
```
.\venv\Scripts\python.exe dashboard\app.py
```
Dashboard will be running at: `http://127.0.0.1:8080`

### Step 5 — Open browser
```
http://127.0.0.1:8080
```

---

## 🔐 API Authentication

All API endpoints (except `/` and `/health`) are protected with an API key.

Include this header in every request:
```
X-API-Key: oxvenom-ids-2026
```

### API Endpoints

| Method | Endpoint | Description | Auth Required |
|---|---|---|---|
| GET | `/` | System info | ❌ |
| GET | `/health` | Health check | ❌ |
| POST | `/predict` | Classify network traffic | ✅ |
| POST | `/train/start` | Start model training | ✅ |
| GET | `/train/logs` | Get training logs | ✅ |
| POST | `/upload` | Upload dataset file | ✅ |

### Example API Call
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: oxvenom-ids-2026" \
  -d '{"duration":0,"protocol_type":"tcp","service":"http","flag":"SF","src_bytes":100,...}'
```

### Example Response
```json
{
  "success": true,
  "result": {
    "prediction": "Normal",
    "attack_probability": 0.0005,
    "confidence": "100.0%",
    "alert": false
  },
  "alert_message": "✅ Traffic is Normal"
}
```

---

## 📊 Dataset — NSL-KDD

The NSL-KDD dataset is an improved version of the KDD Cup 1999 dataset, widely used for evaluating intrusion detection systems.

- **Training set:** 125,973 records
- **Testing set:** 22,544 records
- **Features:** 41 network traffic features
- **Classes:** Normal, DoS, Probe, R2L, U2R

Attack categories detected:
| Category | Description | Examples |
|---|---|---|
| DoS | Denial of Service | neptune, smurf, back |
| Probe | Network Scanning | ipsweep, nmap, portsweep |
| R2L | Remote to Local | ftp_write, guess_passwd |
| U2R | User to Root | buffer_overflow, rootkit |

---

## 👨‍💻 Author

**Chethan**
AI-Based Network Intrusion Detection System
College Project — 2026
