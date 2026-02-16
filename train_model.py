import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

DATA_PATH = os.path.join('dataset', 'creditcard.csv')
MODEL_PATH = 'model.pkl'
SCALER_PATH = 'scaler.pkl'

if not os.path.exists(DATA_PATH):
    print("Dataset not found at dataset/creditcard.csv")
    print("Please download the dataset from Kaggle and place it in the dataset/ folder.")
    exit(1)

print("Loading dataset...")
data = pd.read_csv(DATA_PATH)

# Use Time and Amount for demo; extend with more features as needed
X = data[['Time', 'Amount']]
y = data['Class']

print("Handling imbalance with SMOTE (may take time)...")
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_res)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_res, test_size=0.2, random_state=42)

print("Training RandomForest...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Evaluating...")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

print("Saving model and scaler...")
pickle.dump(model, open(MODEL_PATH, 'wb'))
pickle.dump(scaler, open(SCALER_PATH, 'wb'))

print("Model and scaler saved to model.pkl and scaler.pkl")
