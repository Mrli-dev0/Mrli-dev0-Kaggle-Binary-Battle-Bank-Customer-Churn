import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import torch
import joblib
import os
import warnings
from utils import FastResMLP, feature_engineering_infer, DEVICE

warnings.filterwarnings('ignore')

BASE_DIR = "/home/a/kaggle/binary-battle-ml-bank-customer-churn-challenge"
MODEL_DIR = os.path.join(BASE_DIR, "models")

if __name__ == "__main__":
    test = pd.read_csv("/home/a/kaggle/binary-battle-ml-bank-customer-churn-challenge/test.csv")
    test.columns = test.columns.str.strip()

    imp = joblib.load(os.path.join(MODEL_DIR, "imputer.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    mask = joblib.load(os.path.join(MODEL_DIR, "feature_mask.pkl"))
    te = joblib.load(os.path.join(MODEL_DIR, "target_encoder.pkl"))
    le_sur = joblib.load(os.path.join(MODEL_DIR, "le_surname.pkl"))

    X_test = feature_engineering_infer(test, te, le_sur)
    X_test = imp.transform(X_test)
    X_test = X_test[:, mask]
    Xt_s = scaler.transform(X_test)

    pred = np.zeros(len(X_test))
    for fold in range(10):
        m_lgb = joblib.load(os.path.join(MODEL_DIR, f"lgb_{fold}.pkl"))
        m_xgb = joblib.load(os.path.join(MODEL_DIR, f"xgb_{fold}.pkl"))
        m_cat = CatBoostClassifier()
        m_cat.load_model(os.path.join(MODEL_DIR, f"cat_{fold}.cbm"))

        m_mlp = FastResMLP(X_test.shape[1]).to(DEVICE)
        m_mlp.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"mlp_{fold}.pth"), map_location=DEVICE))
        m_mlp.eval()

        p1 = m_lgb.predict_proba(X_test)[:, 1]
        p2 = m_xgb.predict_proba(X_test)[:, 1]
        p3 = m_cat.predict_proba(X_test)[:, 1]

        with torch.no_grad():
            p4 = torch.sigmoid(m_mlp(torch.tensor(Xt_s, dtype=torch.float32).to(DEVICE))).cpu().numpy()

        pred += (p1 + p2 + p3 + p4) / 4 / 10

    sub = pd.DataFrame({"id": test.id, "Exited": pred})
    sub.to_csv(os.path.join(BASE_DIR, "submission.csv"), index=False)