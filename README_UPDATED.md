# Credit Card Fraud Detection (Flask + ML)

This repository contains a beginner-to-intermediate level Credit Card Fraud Detection web application built with Flask and Scikit-learn.

Features
- Real-time fraud prediction via a web form
- Dashboard showing counts of Fraud vs Genuine
- Training script with SMOTE + RandomForest
- SQLite DB to store transaction history

Quick start
1. Create virtualenv and install deps
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Place dataset
- Download the Kaggle Credit Card Fraud dataset and place it at `dataset/creditcard.csv`.

3. Train model (optional but recommended)
```bash
python train_model.py
```
This will produce `model.pkl` and `scaler.pkl` in the project root.

4. Run the app
```bash
python app.py
```
Open http://127.0.0.1:5000

If you don't train a model, the app uses a simple heuristic fallback for demo purposes.

Repository layout
```
/workspaces/credit-card-fraud-detection-ml
├─ app.py
├─ train_model.py
├─ requirements.txt
├─ templates/
└─ static/
```

Next steps I can help with:
- Add pinned dependency versions to `requirements.txt`
- Add a small sample CSV for quick demo
- Run a quick syntax check / smoke test
- Create a GitHub repo and push (I will give exact commands)

Tell me what you'd like me to do next.