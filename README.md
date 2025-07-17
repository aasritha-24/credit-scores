# DeFi Wallet Credit Scoring using Machine Learning

## Overview

This project implements a machine learning-based credit scoring system for wallets interacting with the Aave V2 DeFi protocol.  
Each wallet is assigned a credit score between 0 and 1000 based on historical transaction behavior, where a higher score indicates more reliable and responsible usage.

---

## Table of Contents

- [Introduction](#introduction)  
- [Data and Features](#data-and-features)  
- [Model Architecture](#model-architecture)  
- [Processing Flow](#processing-flow)  
- [Scoring Methodology](#scoring-methodology)  
- [How to Run](#how-to-run)  
- [Future Improvements](#future-improvements)  

---

## Introduction

DeFi lending platforms require reliable mechanisms to evaluate the creditworthiness of wallets. This system uses raw transaction data from the Aave V2 protocol to train an XGBoost classifier that predicts liquidation risk. The predicted probabilities are used to compute credit scores.

---

## Data and Features

- **Input:** JSON file with transaction-level data per wallet including actions like deposit, borrow, repay, liquidationcall, and associated amounts and timestamps.

- **Features Engineered:**  
  - Counts of each action per wallet (deposit, borrow, repay, liquidation).  
  - Total amounts deposited, borrowed, and repaid.  
  - Account age in days based on transaction timestamps.  
  - Transaction frequency (`tx_per_day`).  
  - Ratios: borrow-to-deposit, repay-to-borrow.  
  - Log-transformed amount features to reduce skewness.

- **Label:** Binary indicator whether the wallet ever experienced a liquidation event (`is_liquidated`).

---

## Model Architecture

- **Model Used:** XGBoost classifier (`XGBClassifier`) set for binary logistic regression.  
- **Class Imbalance Handling:** Using `scale_pos_weight` calculated from train target distribution.  
- **Metrics:** Area under ROC curve (`auc`) used during training.

---

## Processing Flow

1. **Load and flatten JSON transaction data** into pandas DataFrame.  
2. **Preprocess**: convert amounts to numeric and timestamps to datetime.  
3. **Aggregate features** by wallet.  
4. **Split data** into stratified train and test sets.  
5. **Train XGBoost model** with class imbalance compensation.  
6. **Predict liquidation risk probabilities** on full dataset.  
7. **Assign credit scores** by ranking and scaling risk probabilities from 0 (highest risk) to 1000 (lowest risk).  
8. **Export scored wallets** to CSV.

---

## Scoring Methodology

- Model's predicted liquidation probability (`risk_prob`) indicates wallet risk.  
- Wallets are ranked by their `risk_prob` values.  
- The rank is normalized and inverted such that lower risk gets higher scores.  
- Scores scaled linearly to 0-1000 integer range:
  ```bash
  credit_score = int((1 - normalized_rank) * 1000)
  ```
---

## How to Run

1. Ensure you have dependencies installed:
```bash
pip install pandas numpy scikit-learn xgboost matplotlib
```

2. Place your transaction JSON file as `user-wallet-transactions.json` in the working directory.

3. Run the main script `wallet_credit_scoring.py` (the provided Python code).

4. Output file `wallet_credit_scores.csv` will contain wallet addresses and their credit scores.

---

## Future Improvements

- Enrich features with temporal patterns, wallet connections, and multi-protocol activity.  
- Apply model calibration for better probability estimates.  
- Explore alternative or ensemble models for performance gains.  
- Incorporate explainability tools such as SHAP for transparency.
