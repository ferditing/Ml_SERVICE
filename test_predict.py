# test_predict.py
# Quick sanity test: loads artifact and prints feature list + does sample prediction.

import joblib
import os
import numpy as np
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(__file__), "decision_tree_model.pkl")

def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Run train_decision_tree.py first.")
    art = joblib.load(MODEL_PATH)
    model = art.get("model") if isinstance(art, dict) else art
    features = art.get("features", []) if isinstance(art, dict) else []
    labels = art.get("label_encoder_classes", []) if isinstance(art, dict) else []

    print("Loaded artifact. Features count:", len(features))
    print("Feature names:", features[:50])
    print("Label classes:", labels)

    # Build a zero-valued sample (most safe) and predict
    if features:
        sample = pd.DataFrame([[0]*len(features)], columns=features)
    else:
        # fallback single zero
        sample = pd.DataFrame([[0]])
    pred = model.predict(sample)
    proba = model.predict_proba(sample) if hasattr(model, "predict_proba") else None
    print("Predicted index:", int(pred[0]))
    if labels:
        print("Predicted label:", labels[int(pred[0])])
    if proba is not None:
        print("Top confidence:", float(proba.max()))
        print("Raw proba:", proba.tolist())

if __name__ == "__main__":
    main()