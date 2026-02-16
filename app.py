from flask import Flask, render_template, request, redirect, url_for
import sqlite3
import os
import pickle
import numpy as np

app = Flask(__name__)
DB_PATH = "database.db"
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

# Load model & scaler if available
model = None
scaler = None
try:
    if os.path.exists(MODEL_PATH):
        model = pickle.load(open(MODEL_PATH, "rb"))
    if os.path.exists(SCALER_PATH):
        scaler = pickle.load(open(SCALER_PATH, "rb"))
except Exception:
    model = None
    scaler = None

# DB init
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS transactions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, time REAL, amount REAL, result TEXT, probability REAL)''')
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT result, COUNT(*) FROM transactions GROUP BY result")
    rows = c.fetchall()
    conn.close()

    fraud = 0
    genuine = 0
    for r in rows:
        if r[0] == 'Fraud':
            fraud = r[1]
        else:
            genuine = r[1]

    return render_template('dashboard.html', fraud=fraud, genuine=genuine)

@app.route('/predict_page')
def predict_page():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        time_val = float(request.form.get('time', 0))
        amount = float(request.form.get('amount', 0))
    except ValueError:
        return redirect(url_for('predict_page'))

    features = np.array([[time_val, amount]])
    prob = 0.0
    pred_label = 0

    if scaler is not None:
        features = scaler.transform(features)

    if model is not None:
        try:
            if hasattr(model, 'predict_proba'):
                prob = float(model.predict_proba(features)[0][1])
            pred_label = int(model.predict(features)[0])
        except Exception:
            pred_label = 0
            prob = 0.0
    else:
        # Fallback: simple heuristic
        pred_label = 1 if amount > 20000 else 0
        prob = 0.85 if pred_label == 1 else 0.05

    result = 'Fraud' if pred_label == 1 else 'Genuine'
    risk = 'High' if prob >= 0.7 else ('Medium' if prob >= 0.3 else 'Low')

    # save to db
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO transactions (time, amount, result, probability) VALUES (?,?,?,?)',
              (time_val, amount, result, float(prob)))
    conn.commit()
    conn.close()

    return render_template('result.html', result=result, risk=risk, probability=round(prob*100, 2))

if __name__ == '__main__':
    app.run(debug=True)
