import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.feature_selection import mutual_info_classif
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import joblib
import warnings
from utils import (
    FastResMLP, feature_engineering_dask_fast,
    LGB_PARAMS, XGB_PARAMS, CAT_PARAMS, SEED, DEVICE
)

warnings.filterwarnings('ignore')

os.environ["MKL_NUM_THREADS"] = "14"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "14"
os.environ["XGBOOST_THREAD"] = "1"

BASE_DIR = "/home/a/kaggle/binary-battle-ml-bank-customer-churn-challenge"
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def train_mlp_single(xtr, ytr, input_dim):
    model = FastResMLP(input_dim).to(DEVICE)
    pos = torch.tensor([(ytr == 0).sum() / (ytr == 1).sum()], dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos)
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    Xtr = torch.tensor(xtr, dtype=torch.float32).to(DEVICE)
    Ytr = torch.tensor(ytr, dtype=torch.float32).to(DEVICE)
    ds = TensorDataset(Xtr, Ytr)
    loader = DataLoader(ds, batch_size=512, shuffle=True, num_workers=0)

    for _ in range(25):
        model.train()
        for bx, by in loader:
            opt.zero_grad()
            criterion(model(bx), by).backward()
            opt.step()
    return model

if __name__ == "__main__":
    train = pd.read_csv("/home/a/kaggle/binary-battle-ml-bank-customer-churn-challenge/train.csv")
    test = pd.read_csv("/home/a/kaggle/binary-battle-ml-bank-customer-churn-challenge/test.csv")
    train.columns = train.columns.str.strip()
    test.columns = test.columns.str.strip()

    X, Xt, y, te, le_sur = feature_engineering_dask_fast(train, test)

    imp = KNNImputer(n_neighbors=8)
    X = imp.fit_transform(X)
    Xt = imp.transform(Xt)

    mi = mutual_info_classif(X, y, random_state=SEED)
    mask = mi > np.percentile(mi, 20)
    X_selected = X[:, mask]

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_selected)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    for fold, (tr, val) in enumerate(skf.split(X_selected, y)):
        xt, xv = X_selected[tr], X_selected[val]
        yt, yv = y[tr], y[val]
        xs_t = X_s[tr]

        m_lgb = lgb.LGBMClassifier(**LGB_PARAMS, n_estimators=25000, device='cuda', n_jobs=14, verbose=-1)
        m_lgb.fit(xt, yt, eval_set=[(xv, yv)], callbacks=[lgb.early_stopping(200, verbose=False)])

        m_xgb = xgb.XGBClassifier(**XGB_PARAMS, n_estimators=25000, tree_method='hist', device='cuda', nthread=1, verbosity=0, early_stopping_rounds=200)
        m_xgb.fit(xt, yt, eval_set=[(xv, yv)], verbose=False)

        m_cat = CatBoostClassifier(**CAT_PARAMS, iterations=25000, task_type='GPU', verbose=False, early_stopping_rounds=200)
        m_cat.fit(xt, yt, eval_set=(xv, yv), use_best_model=True)

        m_mlp = train_mlp_single(xs_t, yt, xt.shape[1])

        joblib.dump(m_lgb, os.path.join(MODEL_DIR, f"lgb_{fold}.pkl"))
        joblib.dump(m_xgb, os.path.join(MODEL_DIR, f"xgb_{fold}.pkl"))
        m_cat.save_model(os.path.join(MODEL_DIR, f"cat_{fold}.cbm"))
        torch.save(m_mlp.state_dict(), os.path.join(MODEL_DIR, f"mlp_{fold}.pth"))

    joblib.dump(imp, os.path.join(MODEL_DIR, "imputer.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(mask, os.path.join(MODEL_DIR, "feature_mask.pkl"))
    joblib.dump(te, os.path.join(MODEL_DIR, "target_encoder.pkl"))
    joblib.dump(le_sur, os.path.join(MODEL_DIR, "le_surname.pkl"))