from features import normalize
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "data/landmarks/landmark_data.csv"
MODEL_PATH = "models/asl_model.pkl"
os.makedirs("models", exist_ok=True)

df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} samples across {df['label'].nunique()} classes")

X = normalize(df.drop("label", axis=1).values)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"\nAccuracy: {acc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, preds))

joblib.dump(model, MODEL_PATH)
print(f"\nModel saved to {MODEL_PATH}")

# Confusion matrix plot (great for your report)
cm = confusion_matrix(y_test, preds, labels=model.classes_)
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=False, xticklabels=model.classes_, yticklabels=model.classes_, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("ASL Classifier Confusion Matrix")
plt.tight_layout()
plt.savefig("models/confusion_matrix.png")
print("Confusion matrix saved to models/confusion_matrix.png")
plt.show()