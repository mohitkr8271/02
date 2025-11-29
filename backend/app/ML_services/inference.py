import os
import joblib
import numpy as np
import lightgbm as lgb
import pandas as pd

# ----------------------------------------------------
# Artifact Folder Path (REQUIRED for Render)
# ----------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

# ----------------------------------------------------
# Load Artifacts
# ----------------------------------------------------
try:
    transformer = joblib.load(os.path.join(ARTIFACTS_DIR, "transformer.joblib"))
    lr_model = joblib.load(os.path.join(ARTIFACTS_DIR, "logistic.pkl"))
    iso_model = joblib.load(os.path.join(ARTIFACTS_DIR, "isotonic.joblib"))
    lgb_model = lgb.Booster(model_file=os.path.join(ARTIFACTS_DIR, "lightgbm.txt"))

except Exception as e:
    raise RuntimeError(f"Error loading ML artifacts: {e}")

# ----------------------------------------------------
# Prediction Function
# ----------------------------------------------------
def predict_model(features: dict):
    """
    Main prediction function.
    `features` must be a dict like:
    {
        "age": 27,
        "salary": 45000,
        "employment_type": "salaried",
        "loan_amount": 200000
    }
    """

    # Safety check
    if not isinstance(features, dict):
        raise ValueError("Expected input features as a dictionary.")

    # Convert dict → DataFrame
    X = pd.DataFrame([features])

    # --------------------------
    # Apply preprocessing
    # --------------------------
    try:
        Xt = transformer.transform(X)
    except Exception as e:
        raise ValueError(f"Preprocessing failed: {e}")

    # --------------------------
    # Logistic + Isotonic
    # --------------------------
    try:
        lr_proba = lr_model.predict_proba(Xt)[:, 1]
        calibrated_lr = iso_model.predict(lr_proba)
    except Exception as e:
        raise ValueError(f"Logistic model failed: {e}")

    # --------------------------
    # LightGBM Probability
    # --------------------------
    try:
        lgb_proba = float(lgb_model.predict(Xt)[0])
    except Exception as e:
        raise ValueError(f"LightGBM model failed: {e}")

    # --------------------------
    # Final Probability
    # --------------------------
    final_proba = float((calibrated_lr + lgb_proba) / 2)

    approved = final_proba >= 0.5

    return {
        "probability": final_proba,
        "approved": approved,
        "decision": "Approved" if approved else "Rejected"
    }
