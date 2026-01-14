from fastapi import FastAPI, HTTPException
import joblib
import os

from preprocessing import build_feature_vector

# ----------------------------
# App
# ----------------------------
app = FastAPI(title="Smart Livestock ML Service")

# ----------------------------
# Load model artifact
# ----------------------------
ROOT = os.path.dirname(__file__)
MODEL_PATH = os.path.join(ROOT, "decision_tree_model.pkl")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model not found at {MODEL_PATH}")

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
FEATURES = artifact["features"]
LABELS = artifact["label_encoder_classes"]

print(f"Loaded model with {len(FEATURES)} features")
print("Labels:", LABELS)

# ----------------------------
# Health check
# ----------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "features_count": len(FEATURES),
        "labels": LABELS
    }

# ----------------------------
# Prediction endpoint
# ----------------------------
@app.post("/predict")
def predict(payload: dict):
    try:
        X = build_feature_vector(payload, FEATURES)
        pred_idx = int(model.predict(X)[0])
        proba = float(model.predict_proba(X).max())

        return {
            "predicted_label": LABELS[pred_idx],
            "confidence": round(proba, 3)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
