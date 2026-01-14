import re
import numpy as np

def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def build_feature_vector(payload: dict, FEATURES: list[str]) -> np.ndarray:
    """
    Converts raw farmer report into model-ready feature vector.
    """

    # initialize all features to 0
    vec = {f: 0 for f in FEATURES}

    # numeric fields
    vec["age"] = payload.get("age", 0) or 0
    vec["body_temperature"] = payload.get("body_temperature", 0) or 0

    # animal one-hot
    animal = payload.get("animal_type", "")
    animal_key = f"animal_{animal.lower()}"
    if animal_key in vec:
        vec[animal_key] = 1

    # symptoms → binary columns
    symptoms = payload.get("symptoms", [])
    for s in symptoms:
        s_norm = normalize(s)
        col = s_norm.replace(" ", "_")
        if col in vec:
            vec[col] = 1

    # return in correct column order
    return np.array([[vec[f] for f in FEATURES]])
