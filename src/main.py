import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier


# Create folders if missing
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

print("Loading dataset...")

# Load dataset
df = pd.read_csv("data/dataset.csv")

print("Dataset Loaded Successfully")
print(df.head())


# Clean data
print("\nCleaning data...")

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

print("Data cleaned successfully")


# Encode labels
label_column = ' Label'

encoder = LabelEncoder()

df[label_column] = encoder.fit_transform(df[label_column])

print("Labels encoded")


# Features and target
X = df.drop(label_column, axis=1)
y = df[label_column]


# Split dataset
print("Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Random Forest...")

rf = RandomForestClassifier()

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)

print("Random Forest Accuracy:", rf_accuracy)


print("Training XGBoost...")

xgb = XGBClassifier()

xgb.fit(X_train, y_train)

xgb_pred = xgb.predict(X_test)

xgb_accuracy = accuracy_score(y_test, xgb_pred)

print("XGBoost Accuracy:", xgb_accuracy)


# Accuracy Graph
models = ['Random Forest', 'XGBoost']
accuracy = [rf_accuracy, xgb_accuracy]

plt.figure(figsize=(6, 4))
plt.bar(models, accuracy)
plt.ylabel("Accuracy")
plt.title("Model Comparison")

plt.savefig("outputs/accuracy_graph.png")
plt.show()


# Confusion Matrix
cm = confusion_matrix(y_test, rf_pred)

print("\nConfusion Matrix:")
print(cm)


# Classification Report
report = classification_report(y_test, rf_pred)

print("\nClassification Report:")
print(report)


# Save classification report
with open("outputs/classification_report.txt", "w") as file:
    file.write(report)

print("Classification report saved")


# Save model
joblib.dump(rf, "models/random_forest.pkl")

print("Model saved successfully")

print("\nProject Completed Successfully!")