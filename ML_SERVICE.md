## `README.md` (ML Service – **Plugin Guide**)

```md
# SmartLivestock Connect – ML Diagnostic Service

This repository contains the **machine learning service** used to provide
preliminary livestock disease predictions based on reported symptoms.

The model has been **retrained** and exposed as a REST API for integration
with the backend system.

---

## 🧠 Model Overview

- Algorithm: Decision Tree Classifier
- Input: Text-based symptoms
- Output:
  - Predicted disease
  - Confidence score

This service is **advisory only** and does not replace veterinary diagnosis.

---

## 🚀 Tech Stack

- Python 3.9+
- Flask / FastAPI
- scikit-learn
- joblib
- pandas
- numpy

---

## 📂 Project Structure

```txt
ml_service/
├── model/
│   ├── model.pkl
│   └── vectorizer.pkl
├── app.py
├── requirements.txt
└── README.md
🔧 Setup Instructions
bash
Copy code
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
Service runs on:

arduino
Copy code
http://localhost:5000
🔌 API Endpoint
POST /predict
Request
json
Copy code
{
  "symptoms": "coughing, fever, nasal discharge"
}
Response
json
Copy code
{
  "disease": "Pneumonia",
  "confidence": 0.87
}
🔗 Backend Plugin Procedure
1. Ensure ML service is running
bash
Copy code
python app.py
2. Set backend environment variable
env
Copy code
ML_SERVICE_URL=http://localhost:5000
3. Backend forwards symptom reports to ML service
ts
Copy code
POST /api/ml/predict → ML Service → response returned to frontend
⚠️ Notes
This service must be running before backend predictions work

Designed for local or containerized deployment

Can be Dockerized independently

📌 Related Repositories
Backend API: smartlivestock-backend

Frontend: smartlivestock-frontend

yaml
Copy code
