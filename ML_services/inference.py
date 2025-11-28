import joblib
import numpy as np
import lightgbm as lgb
import json
import os

ARTIFACTS = "artifacts"

# Load artifacts
transformer = joblib.load(os.path.join(ARTIFACTS, "transformer.joblib"))
lr_model = joblib.load(os.path.join(ARTIFACTS, "logistic.pkl"))
iso_model = joblib.load(os.path.join(ARTIFACTS, "isotonic.joblib"))

lgb_model = lgb.Booster(model_file=os.path.join(ARTIFACTS, "lightgbm.txt"))

# ------------------------
# Prediction Function
# ------------------------
def predict(data: dict):

    # Convert dict → DataFrame row
    import pandas as pd
    X = pd.DataFrame([data])

    # Apply preprocessing
    Xt = transformer.transform(X)

    # Logistic + Isotonic
    lr_proba = lr_model.predict_proba(Xt)[:, 1]
    calibrated = iso_model.predict(lr_proba)

    # LightGBM
    lgb_proba = lgb_model.predict(Xt)[0]

    # Final probability (average both)
    final_proba = float((calibrated + lgb_proba) / 2)

    decision = 1 if final_proba >= 0.5 else 0

    return {
        "probability": final_proba,
        "decision": "Approved" if decision == 1 else "Rejected"
    }
