#import the packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)    
#read the dataset
df = pd.read_csv("loan_approval.csv")
print("First 5 Records")
print(df.head())
print("\nDataset Shape")
print(df.shape)
print("\nDataset Information")
print(df.info())
print("\nStatistical Summary")
print(df.describe())
# 3 data preprocessing
print("\nMissing Values")
print(df.isnull().sum())

# Remove duplicate records
df.drop_duplicates(inplace=True)

# Convert categorical columns into numbers
le = LabelEncoder()

df["name"] = le.fit_transform(df["name"])
df["city"] = le.fit_transform(df["city"])

print("\nPreprocessing Completed")

#eda

print("\nLoan Approval Count")
print(df["loan_approved"].value_counts())

print("\nCorrelation Matrix")
print(df.corr(numeric_only=True))
#visualization

#loan approval distribution
df["loan_approved"].value_counts().plot(
    kind="bar",
    color=["blue","pink"]
)
plt.title("Loan Approval Distribution")
plt.xlabel("Loan Approved")
plt.ylabel("Count")
plt.show()
#credit score histogram
plt.figure(figsize=(6,4))
plt.hist(df["credit_score"], bins=20)
plt.title("Credit Score Distribution")
plt.xlabel("Credit Score")
plt.ylabel("Frequency")
plt.show()
#income histogram
plt.figure(figsize=(6,4))
plt.hist(df["income"], bins=20)
plt.title("Income Distribution")
plt.xlabel("Income")
plt.ylabel("Frequency")
plt.show()
# 5.feature slection

X = df.drop("loan_approved", axis=1)
y = df["loan_approved"]
# 6. TRAIN TEST SPLIT
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
# 7. MODEL SELECTION
# -----------------------------

model = LogisticRegression(max_iter=1000)
# 8. MODEL TRAINING
# -----------------------------

model.fit(X_train, y_train)

print("\nModel Training Completed")
# 9. MODEL TESTING
# -----------------------------

y_pred = model.predict(X_test)
# Confusion Matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred
)
plt.title("Confusion Matrix")
plt.show()

# ROC Curve
RocCurveDisplay.from_estimator(
    model,
    X_test,
    y_test
)
plt.title("ROC Curve")
plt.show()
# 10. MODEL EVALUATION
# -----------------------------

print("\nAccuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))
plt.show()
# FEATURE IMPORTANCE
# -----------------------------

coef = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})

print("\nFeature Importance")
print(coef.sort_values(by="Coefficient", ascending=False))
 
joblib.dump(model, "model.pk1")
print("model saved successfully!")



