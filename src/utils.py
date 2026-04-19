import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import dask.dataframe as dd
from sklearn.preprocessing import LabelEncoder
from category_encoders import TargetEncoder

LGB_PARAMS = {
    'learning_rate': 0.012,
    'num_leaves': 192,
    'max_depth': 10,
    'reg_alpha': 0.2,
    'reg_lambda': 0.5,
    'subsample': 0.95,
    'colsample_bytree': 0.95
}

XGB_PARAMS = {
    'learning_rate': 0.012,
    'max_depth': 9,
    'reg_alpha': 0.2,
    'reg_lambda': 0.5,
    'subsample': 0.95,
    'colsample_bytree': 0.95
}

CAT_PARAMS = {
    'learning_rate': 0.014,
    'depth': 8,
    'l2_leaf_reg': 2.0,
    'subsample': 0.95,
    'bootstrap_type': 'Bernoulli'
}

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FastResMLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 128),
            nn.BatchNorm1d(128),
            nn.Mish(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def feature_engineering_dask_fast(train, test):
    df_train = dd.from_pandas(train.copy(), npartitions=14)
    df_test = dd.from_pandas(test.copy(), npartitions=14)
    y_train = train['Exited'].values

    for df in [df_train, df_test]:
        df['Surname'] = df['Surname'].fillna('Unknown')
        df['Surname_Len'] = df['Surname'].str.len()
        df['AgeWhenJoin'] = df['Age'] - df['Tenure']
        df['BalanceZero'] = (df['Balance'] == 0).astype(int)
        df['Is3Prod'] = (df['NumOfProducts'] == 3).astype(int)
        df['Is4Prod'] = (df['NumOfProducts'] == 4).astype(int)
        df['InactiveNoBalance'] = (df['IsActiveMember'] == 0) & (df['Balance'] == 0)
        df['BalSalaryRatio'] = df['Balance'] / (df['EstimatedSalary'] + 1e-8)
        df['AgePerProd'] = df['Age'] / (df['NumOfProducts'] + 1)
        df['BalPerTenure'] = df['Balance'] / (df['Tenure'] + 1)
        df['AgeTenureRatio'] = df['Age'] / (df['Tenure'] + 1)
        df['SalaryPerProd'] = df['EstimatedSalary'] / (df['NumOfProducts'] + 1)
        df['AgeBalRatio'] = df['Age'] / (df['Balance'] + 1e-8)
        df['CreditAge'] = df['CreditScore'] * df['Age']
        df['TenureBal'] = df['Tenure'] * df['Balance']
        df['ActiveProd'] = df['IsActiveMember'] * df['NumOfProducts']
        df['AgeProdActive'] = df['Age'] * df['NumOfProducts'] * df['IsActiveMember']
        df['CreditBal'] = df['CreditScore'] * df['Balance']
        df['TenureAgeBal'] = df['Tenure'] * df['Age'] * df['Balance']
        df['LogBal'] = np.log1p(df['Balance'])
        df['LogSalary'] = np.log1p(df['EstimatedSalary'])
        df['LogBalSalary'] = np.log1p(df['BalSalaryRatio'])
        df['AgeBin'] = df['Age'].map(lambda x: np.digitize(x, bins=[0,25,35,45,55,65,100])-1)
        df['CreditBin'] = df['CreditScore'].map(lambda x: np.digitize(x, bins=[0,400,500,600,700,800,1000])-1)
        for c in ['Age', 'Balance', 'CreditScore', 'EstimatedSalary']:
            q25 = df[c].quantile(0.25)
            q75 = df[c].quantile(0.75)
            df[f'{c}_q25'] = (df[c] < q25).astype(int)
            df[f'{c}_q75'] = (df[c] > q75).astype(int)
        df['HighAgeLowCredit'] = ((df['Age'] > 45) & (df['CreditScore'] < 550)).astype(int)
        df['InactiveHighBal'] = ((df['IsActiveMember'] == 0) & (df['Balance'] > 50000)).astype(int)
        df['ManyProdLowTenure'] = ((df['NumOfProducts'] >= 3) & (df['Tenure'] <= 2)).astype(int)
        df['Geo_Gender'] = df['Geography'] + "_" + df['Gender']
        df['Geo_Prod'] = df['Geography'].astype(str) + "_" + df['NumOfProducts'].astype(str)

    X_train = df_train.compute()
    X_test = df_test.compute()

    le_sur = LabelEncoder()
    X_train['Surname'] = le_sur.fit_transform(X_train['Surname'])
    X_test['Surname'] = X_test['Surname'].apply(lambda x: le_sur.transform([x])[0] if x in le_sur.classes_ else len(le_sur.classes_))

    te_cols = ['Geography', 'Gender', 'Geo_Gender', 'Geo_Prod', 'AgeBin', 'NumOfProducts', 'Surname']
    te = TargetEncoder(cols=te_cols, smoothing=5)
    X_train = pd.concat([X_train, te.fit_transform(X_train[te_cols], y_train).add_suffix("_te")], axis=1)
    X_test = pd.concat([X_test, te.transform(X_test[te_cols]).add_suffix("_te")], axis=1)

    for c in ['Age', 'CreditScore', 'Balance', 'EstimatedSalary']:
        X_train[f'{c}_bin'], b = pd.qcut(X_train[c], 5, duplicates='drop', retbins=True)
        X_test[f'{c}_bin'] = pd.cut(X_test[c], b, labels=False)
        g = X_train.groupby(f'{c}_bin')['Exited'].agg(pos='sum', cnt='count')
        g['neg'] = g['cnt'] - g['pos']
        pos_all = y_train.sum()
        neg_all = len(y_train) - pos_all
        g['woe'] = np.log((g['pos'] / pos_all + 1e-8) / (g['neg'] / neg_all + 1e-8))
        X_train[f'{c}_woe'] = X_train[f'{c}_bin'].map(g['woe'])
        X_test[f'{c}_woe'] = X_test[f'{c}_bin'].map(g['woe']).fillna(0)

    drop = ['RowNumber', 'CustomerId', 'id', 'Exited'] + [c for c in X_train.columns if '_bin' in c]
    X_train = X_train.drop(columns=[c for c in drop if c in X_train.columns], errors='ignore')
    X_test = X_test.drop(columns=[c for c in drop if c in X_test.columns], errors='ignore')

    for c in ['Geography', 'Gender', 'Geo_Gender', 'Geo_Prod']:
        le = LabelEncoder()
        X_train[c] = le.fit_transform(X_train[c])
        X_test[c] = le.transform(X_test[c])

    return X_train, X_test, y_train, te, le_sur

def feature_engineering_infer(test, te, le_sur):
    df = test.copy()
    df['Surname'] = df['Surname'].fillna('Unknown')
    df['Surname_Len'] = df['Surname'].str.len()
    df['AgeWhenJoin'] = df['Age'] - df['Tenure']
    df['BalanceZero'] = (df['Balance'] == 0).astype(int)
    df['Is3Prod'] = (df['NumOfProducts'] == 3).astype(int)
    df['Is4Prod'] = (df['NumOfProducts'] == 4).astype(int)
    df['InactiveNoBalance'] = (df['IsActiveMember'] == 0) & (df['Balance'] == 0)
    df['BalSalaryRatio'] = df['Balance'] / (df['EstimatedSalary'] + 1e-8)
    df['AgePerProd'] = df['Age'] / (df['NumOfProducts'] + 1)
    df['BalPerTenure'] = df['Balance'] / (df['Tenure'] + 1)
    df['AgeTenureRatio'] = df['Age'] / (df['Tenure'] + 1)
    df['SalaryPerProd'] = df['EstimatedSalary'] / (df['NumOfProducts'] + 1)
    df['AgeBalRatio'] = df['Age'] / (df['Balance'] + 1e-8)
    df['CreditAge'] = df['CreditScore'] * df['Age']
    df['TenureBal'] = df['Tenure'] * df['Balance']
    df['ActiveProd'] = df['IsActiveMember'] * df['NumOfProducts']
    df['AgeProdActive'] = df['Age'] * df['NumOfProducts'] * df['IsActiveMember']
    df['CreditBal'] = df['CreditScore'] * df['Balance']
    df['TenureAgeBal'] = df['Tenure'] * df['Age'] * df['Balance']
    df['LogBal'] = np.log1p(df['Balance'])
    df['LogSalary'] = np.log1p(df['EstimatedSalary'])
    df['LogBalSalary'] = np.log1p(df['BalSalaryRatio'])
    df['AgeBin'] = df['Age'].map(lambda x: np.digitize(x, bins=[0,25,35,45,55,65,100])-1)
    df['CreditBin'] = df['CreditScore'].map(lambda x: np.digitize(x, bins=[0,400,500,600,700,800,1000])-1)
    for c in ['Age', 'Balance', 'CreditScore', 'EstimatedSalary']:
        q25 = df[c].quantile(0.25)
        q75 = df[c].quantile(0.75)
        df[f'{c}_q25'] = (df[c] < q25).astype(int)
        df[f'{c}_q75'] = (df[c] > q75).astype(int)
    df['HighAgeLowCredit'] = ((df['Age'] > 45) & (df['CreditScore'] < 550)).astype(int)
    df['InactiveHighBal'] = ((df['IsActiveMember'] == 0) & (df['Balance'] > 50000)).astype(int)
    df['ManyProdLowTenure'] = ((df['NumOfProducts'] >= 3) & (df['Tenure'] <= 2)).astype(int)
    df['Geo_Gender'] = df['Geography'] + "_" + df['Gender']
    df['Geo_Prod'] = df['Geography'].astype(str) + "_" + df['NumOfProducts'].astype(str)

    df['Surname'] = df['Surname'].apply(lambda x: le_sur.transform([x])[0] if x in le_sur.classes_ else len(le_sur.classes_))
    te_df = te.transform(df[te.cols])
    df = pd.concat([df, te_df.add_suffix("_te")], axis=1)

    for c in ['Age', 'CreditScore', 'Balance', 'EstimatedSalary']:
        df[f'{c}_woe'] = 0

    drop = ['RowNumber', 'CustomerId', 'id']
    df = df.drop(columns=[c for c in drop if c in df.columns], errors='ignore')

    for c in ['Geography', 'Gender', 'Geo_Gender', 'Geo_Prod']:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c])

    return df