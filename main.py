import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import json
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
with open('user-wallet-transactions.json', 'r') as f:
    data = json.load(f)

df = pd.json_normalize(data)

df['actionData.amount'] = pd.to_numeric(df['actionData.amount'], errors='coerce')
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

df['actionData.amount'] = pd.to_numeric(df['actionData.amount'], errors='coerce').fillna(0)

grouped = df.groupby('userWallet')
deposit_sum = df[df['action'] == 'deposit'].groupby('userWallet')['actionData.amount'].sum()
borrow_sum = df[df['action'] == 'borrow'].groupby('userWallet')['actionData.amount'].sum()
repay_sum = df[df['action'] == 'repay'].groupby('userWallet')['actionData.amount'].sum()

features = grouped['action'].agg([
    ('deposit_count', lambda x: (x == 'deposit').sum()),
    ('borrow_count', lambda x: (x == 'borrow').sum()),
    ('repay_count', lambda x: (x == 'repay').sum()),
    ('liquidation_count', lambda x: (x == 'liquidationcall').sum()),
    ('total_tx_count', 'count')
]).reset_index()

features = features.merge(deposit_sum.rename('total_deposit_amount'), on='userWallet', how='left')
features = features.merge(borrow_sum.rename('total_borrow_amount'), on='userWallet', how='left')
features = features.merge(repay_sum.rename('total_repay_amount'), on='userWallet', how='left')

features['total_deposit_amount'] = features['total_deposit_amount'].fillna(0)
features['total_borrow_amount'] = features['total_borrow_amount'].fillna(0)
features['total_repay_amount'] = features['total_repay_amount'].fillna(0)

first_tx_time = grouped['timestamp'].min().reset_index(name='first_tx_time')
last_tx_time = grouped['timestamp'].max().reset_index(name='last_tx_time')

features = features.merge(first_tx_time, on='userWallet')
features = features.merge(last_tx_time, on='userWallet')

features['account_age_days'] = (features['last_tx_time'] - features['first_tx_time']).dt.total_seconds() / 86400.0

features['borrow_to_deposit_ratio'] = features['total_borrow_amount'] / (features['total_deposit_amount'] + 1e-9)
features['repay_to_borrow_ratio'] = features['total_repay_amount'] / (features['total_borrow_amount'] + 1e-9)
features['tx_per_day'] = features['total_tx_count'] / (features['account_age_days'] + 1e-9)

for col in ['total_deposit_amount', 'total_borrow_amount', 'total_repay_amount']:
    features[col+'_log'] = np.log1p(features[col])

features['is_liquidated'] = (features['liquidation_count'] > 0).astype(int)

X = features.drop(['userWallet', 'is_liquidated', 'first_tx_time', 'last_tx_time', 'liquidation_count'], axis=1)
y = features['is_liquidated']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

imbalance_ratio = (len(y_train) - sum(y_train)) / sum(y_train)

model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    scale_pos_weight=imbalance_ratio,
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    use_label_encoder=False,
    random_state=42
)
model.fit(X_train, y_train)

y_proba = model.predict_proba(X_test)[:, 1]
threshold = 0.3
y_pred = (y_proba > threshold).astype(int)
cm = confusion_matrix(y_test, y_pred)

features['risk_prob'] = model.predict_proba(X)[:, 1]
gamma = 3
features['credit_score'] = (((1 - features['risk_prob']) ** gamma) * 1000).astype(int)
features[['userWallet', 'credit_score']].to_csv("wallet_credit_scores.csv", index=False)
print(features[['userWallet', 'credit_score']])
