"""
Diabetes Prediction System
Author: Prince
Description: Machine Learning model to predict diabetes risk using medical attributes.
"""

# ============================================================
# DIABETES PREDICTION SYSTEM - FINAL PROFESSIONAL VERSION
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.pipeline import Pipeline

# ------------------------------------------------------------
# 1️⃣ LOAD DATA
# ------------------------------------------------------------

print("Loading dataset...")
df = pd.read_csv("diabetes_prediction_dataset.csv")
df.columns = df.columns.str.strip()

# ------------------------------------------------------------
# 2️⃣ DATA CLEANING & PREPROCESSING
# ------------------------------------------------------------

# Encode gender
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

# Fill binary columns
binary_cols = ['hypertension', 'heart_disease', 'diabetes']
for col in binary_cols:
    df[col] = df[col].fillna(0).astype(int)

# Fill numeric columns
numeric_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Handle smoking history
df['smoking_history'] = df['smoking_history'].fillna('No Info')
df = pd.get_dummies(df, columns=['smoking_history'], drop_first=True)

df = df.fillna(0)

# Remove unrealistic values
df = df[(df['age'] >= 0) & (df['age'] <= 120)]
df = df[(df['bmi'] >= 10) & (df['bmi'] <= 60)]
df = df[(df['HbA1c_level'] >= 4.0) & (df['HbA1c_level'] <= 14.0)]
df = df[(df['blood_glucose_level'] >= 50) & (df['blood_glucose_level'] <= 300)]

# ------------------------------------------------------------
# 3️⃣ FEATURES & TARGET
# ------------------------------------------------------------

X = df.drop('diabetes', axis=1)
y = df['diabetes']

# ------------------------------------------------------------
# 4️⃣ TRAIN-TEST SPLIT (STRATIFIED)
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------------------------------------
# 5️⃣ SCALING (NO DATA LEAKAGE)
# ------------------------------------------------------------

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------
# 6️⃣ MODEL TRAINING (IMBALANCE HANDLING)
# ------------------------------------------------------------

model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_scaled, y_train)

print("\nModel trained successfully!")

# ------------------------------------------------------------
# 7️⃣ CROSS VALIDATION (PROPER - USING PIPELINE)
# ------------------------------------------------------------

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000, class_weight='balanced'))
])

cv_scores = cross_val_score(pipeline, X, y, cv=5)

print("Cross Validation Accuracy:", round(cv_scores.mean(), 4))

# ------------------------------------------------------------
# 8️⃣ MODEL EVALUATION
# ------------------------------------------------------------

y_pred = model.predict(X_test_scaled)

print("\n===== MODEL EVALUATION =====")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ------------------------------------------------------------
# 9️⃣ ROC CURVE
# ------------------------------------------------------------

y_prob = model.predict_proba(X_test_scaled)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()

# ------------------------------------------------------------
# 🔟 FEATURE IMPORTANCE
# ------------------------------------------------------------

importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
})

importance['Absolute Coefficient'] = importance['Coefficient'].abs()
importance = importance.sort_values(by='Absolute Coefficient', ascending=False)

print("\nTop Important Features:")
print(importance[['Feature', 'Coefficient']].head(10))

# ============================================================
# 1️⃣1️⃣ USER PREDICTION SYSTEM
# ============================================================

print("\nENTER PATIENT DETAILS FOR PREDICTION\n")

while True:
    gender = input("Gender (Male/Female or exit): ").strip().lower()

    if gender == "exit":
        print("Exiting prediction system.")
        break

    if gender not in ["male", "female"]:
        print("Invalid gender. Please enter Male or Female.")
        continue

    gender_val = 1 if gender == "male" else 0

    try:
        age = float(input("Age: "))
        hypertension = int(input("Hypertension (0/1): "))
        heart_disease = int(input("Heart Disease (0/1): "))
        bmi = float(input("BMI: "))
        hba1c = float(input("HbA1c Level: "))
        glucose = float(input("Blood Glucose Level: "))
    except ValueError:
        print("Invalid numeric input. Try again.")
        continue

    smoking_cols = [col for col in X.columns if "smoking_history" in col]
    smoking_data = dict.fromkeys(smoking_cols, 0)

    smoking = input("Smoking History (never/former/current/No Info): ").lower()
    key = f"smoking_history_{smoking}"

    if key in smoking_data:
        smoking_data[key] = 1

    input_data = {
        'gender': gender_val,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'bmi': bmi,
        'HbA1c_level': hba1c,
        'blood_glucose_level': glucose
    }

    input_data.update(smoking_data)

    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    print("\nRESULT:")
    if prediction == 1:
        print("⚠️ High Risk of Diabetes")
    else:
        print("✅ Low Risk of Diabetes")

    print(f"Prediction Probability: {probability:.2f}")
    print("-" * 50)
