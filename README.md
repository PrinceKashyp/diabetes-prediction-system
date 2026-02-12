# 🩺 Diabetes Prediction System

## 📌 Project Overview

This project is a Machine Learning-based Diabetes Prediction System developed to predict whether a patient is at risk of diabetes using medical attributes such as:

- Age  
- BMI  
- HbA1c Level  
- Blood Glucose Level  
- Hypertension  
- Heart Disease  
- Gender  
- Smoking History  

The project implements a complete machine learning workflow including preprocessing, model training, evaluation, and an interactive user prediction system.

---

## 📊 Dataset Information

The dataset contains medical and demographic information with the following features:

- Gender  
- Age  
- Hypertension  
- Heart Disease  
- Smoking History  
- BMI  
- HbA1c Level  
- Blood Glucose Level  
- Diabetes (Target Variable)  

---

## ⚙️ Machine Learning Approach

The following steps were performed:

- Data cleaning and preprocessing  
- Encoding categorical variables  
- Feature scaling using `StandardScaler`  
- Stratified train-test split  
- Logistic Regression model training  
- Handling class imbalance using `class_weight='balanced'`  
- Cross-validation  
- Model evaluation using:
  - Accuracy
  - Confusion Matrix
  - Classification Report
  - ROC-AUC Score  

---

## 📈 Model Performance

- Cross Validation Accuracy: ~88%  
- Test Accuracy: ~88–89%  
- Improved recall for diabetic patients after handling class imbalance  
- Strong ROC-AUC score  

The model focuses on reducing false negatives to better identify high-risk diabetic patients.

---

## 🔍 Important Features

Top influential features identified:

- HbA1c Level  
- Blood Glucose Level  
- Age  
- BMI  

These features show strong correlation with diabetes risk.

---

## 💻 User Prediction System

The project includes an interactive command-line interface that allows users to:

- Enter patient details  
- Get diabetes risk prediction  
- View probability score  

---

## 🚀 How to Run the Project

1️⃣ Clone the repository:

```bash
git clone https://github.com/PrinceKashyp/diabetes-prediction-system.git
```

2️⃣ Navigate to project folder:

```bash
cd diabetes-prediction-system
```

3️⃣ Install dependencies:

```bash
pip install -r requirements.txt
```

4️⃣ Run the project:

```bash
python diabetes_prediction.py
```

---

## 📌 Future Improvements

- Model comparison with Random Forest / XGBoost  
- Hyperparameter tuning  
- Web deployment using Streamlit  
- Model deployment with Flask or FastAPI  

---

## 🛠 Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  

---

## 👨‍💻 Author

Prince  
AI & Generative AI Intern  
YBI Foundation
