# 🚀 Bank Loan Approval Prediction

A complete Machine Learning pipeline to predict loan approval status, including data preprocessing, feature engineering, baseline models, hyperparameter tuning, model comparison, evaluation metrics, and visualization reports.

This project implements multiple ML algorithms (Logistic Regression, Random Forest, Gradient Boosting, Extra Trees, Decision Tree, KNN, Naive Bayes, AdaBoost) and performs RandomizedSearchCV-based hyperparameter optimization.
Advanced plots such as model comparison graphs, hyperparameter analysis, and confusion matrices are also generated.

---

## 📌 Features

✔ Complete data preprocessing
✔ Handling missing values
✔ Manual + automated label encoding
✔ Feature engineering
✔ Baseline model training
✔ Hyperparameter tuning using RandomizedSearchCV
✔ Model evaluation using Accuracy and ROC-AUC
✔ Comparison across all models
✔ Visualization & saved PNG reports
✔ Easy to extend for new datasets

---

## 🧠 ML Models Used
🔹 Tuned Models (using RandomizedSearchCV)

Random Forest

Gradient Boosting

Extra Trees Classifier

Logistic Regression

🔹 Baseline Models (no tuning)

Decision Tree

AdaBoost

K-Nearest Neighbors

Naive Bayes

---

## 🧾 Workflow Explanation

### 1️⃣ Load & Preprocess Data

Removes whitespaces

Handles missing numeric & categorical values

Label encodes categorical columns

Feature engineering added:

Loan-to-income ratio

Total assets

Asset-to-loan ratio

Debt burden

### 2️⃣ Train-Test Split

Data is split into X_train, X_test, y_train, y_test.

### 3️⃣ Baseline Models

Each baseline model is trained and evaluated using:

Accuracy

Cross-validation ROC-AUC

### 4️⃣ Hyperparameter Tuning

Performed using RandomizedSearchCV on:

Random Forest

Gradient Boosting

Extra Trees

Logistic Regression

Each tuned model stores:

Best parameters

Best CV ROC-AUC

Trained best estimator

### 5️⃣ Final Evaluation

All models (tuned + baseline) are evaluated on test data:

Accuracy

ROC-AUC (if available)

CV ROC-AUC

### 6️⃣ Visualizations Generated

The script automatically generates:

#### 📌 Model Comparison Plot

Test accuracy

CV ROC-AUC

Test ROC-AUC

Tuned vs Baseline distribution

#### 📌 Hyperparameter Analysis Plot

Best CV scores

n_estimators comparison

max_depth

learning_rate

#### 📌 Confusion Matrix Comparison
Top 4 models based on CV AUC

All saved as:

model_comparison.png

hyperparameter_analysis.png

confusion_matrices.png

---

## ▶️ How to Run

1. Install Dependencies
   pip install pandas numpy scikit-learn seaborn matplotlib scipy
2. Run the Script
   python main_hypertuned.py

---

## 📝 Dataset Requirements

Your dataset must contain these columns for full functionality:

Column Name Description
loan_status Target variable
loan_amount Loan amount requested
income_annum Annual income
loan_term Loan duration
education Applicant education level
self_employed Yes/No
residential_assets_value Financial asset
commercial_assets_value Financial asset
luxury_assets_value Financial asset
bank_asset_value Financial asset

Extra columns are supported automatically.

---

## 🏆 Best Model Selection

The script automatically compares:

Test accuracy

CV ROC-AUC

Test ROC-AUC

And identifies the strongest model for deployment.

---

## 🧑‍💻 Author

Aryan Sengar
B.Tech CSE | ML Developer | Data Enthusiast
📍 India
