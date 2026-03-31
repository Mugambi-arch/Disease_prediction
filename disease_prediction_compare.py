import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score
)

# =========================
# 1. LOAD DATASET
# =========================
file_path = "Training.csv"

df = pd.read_csv(file_path)

# Remove empty/useless columns like Unnamed: 133
df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]

# Fill missing values
df = df.fillna(0)

print("Dataset Shape:", df.shape)
print("Missing values remaining:", df.isnull().sum().sum())

# =========================
# 2. TARGET COLUMN
# =========================
if "prognosis" in df.columns:
    target_col = "prognosis"
elif "disease" in df.columns:
    target_col = "disease"
else:
    raise ValueError("Target column not found. Expected 'prognosis' or 'disease'.")

# =========================
# 3. FEATURES + LABEL
# =========================
X = df.drop(columns=[target_col])
y = df[target_col]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("Number of features:", X.shape[1])
print("Number of disease classes:", len(label_encoder.classes_))

# =========================
# 4. SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# =========================
# 5. MODELS
# =========================
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_split=4),
    "Naive Bayes": BernoulliNB()
}

results = {}

# =========================
# 6. TRAIN + TEST + CROSS VALIDATION
# =========================
for name, model in models.items():
    print("\n" + "=" * 60)
    print("MODEL:", name)
    print("=" * 60)

    # Train
    model.fit(X_train, y_train)

    # Test predictions
    pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, pred, average="weighted", zero_division=0)

    # Cross-validation on full dataset
    cv_scores = cross_val_score(model, X, y_encoded, cv=5)

    results[name] = {
        "model": model,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "cv_scores": cv_scores,
        "cv_mean": cv_scores.mean(),
        "predictions": pred
    }

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Cross Validation Scores:", np.round(cv_scores, 4))
    print(f"Average CV Accuracy: {cv_scores.mean():.4f}\n")

    print("Classification Report:")
    print(classification_report(
        y_test,
        pred,
        target_names=label_encoder.classes_,
        zero_division=0
    ))

# =========================
# 7. BEST MODEL
# =========================
# Use cross-validation mean as the main deciding factor
best_model_name = max(results, key=lambda k: results[k]["cv_mean"])
best_model = results[best_model_name]["model"]
best_pred = results[best_model_name]["predictions"]

print("\n" + "=" * 60)
print("BEST MODEL:", best_model_name)
print(f"Best CV Accuracy: {results[best_model_name]['cv_mean']:.4f}")
print("=" * 60)

# =========================
# 8. SAVE MODEL
# =========================
joblib.dump(best_model, "disease_model.pkl")
print("Saved model as disease_model.pkl")

# =========================
# 9. SAVE RESULTS TABLE
# =========================
results_df = pd.DataFrame({
    "Model": list(results.keys()),
    "Test Accuracy": [results[n]["accuracy"] for n in results],
    "Precision": [results[n]["precision"] for n in results],
    "Recall": [results[n]["recall"] for n in results],
    "F1 Score": [results[n]["f1_score"] for n in results],
    "CV Mean Accuracy": [results[n]["cv_mean"] for n in results]
})
results_df.to_csv("model_results.csv", index=False)
print("Saved model comparison table as model_results.csv")

# =========================
# 10. ACCURACY GRAPH
# =========================
names = list(results.keys())
cv_accs = [results[n]["cv_mean"] for n in names]

plt.figure(figsize=(8, 5))
plt.bar(names, cv_accs)
plt.title("Model Cross-Validation Accuracy Comparison")
plt.ylabel("Average CV Accuracy")
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig("accuracy_chart.png")
plt.show()

# =========================
# 11. CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_test, best_pred)

fig, ax = plt.subplots(figsize=(12, 12))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=label_encoder.classes_
)
disp.plot(ax=ax, xticks_rotation=90)
plt.title(f"Confusion Matrix - {best_model_name}")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# =========================
# 12. FEATURE IMPORTANCE
# =========================
if best_model_name == "Decision Tree":
    importance = best_model.feature_importances_
    idx = np.argsort(importance)[-10:]

    plt.figure(figsize=(8, 6))
    plt.barh(range(len(idx)), importance[idx])
    plt.yticks(range(len(idx)), X.columns[idx])
    plt.title("Top 10 Important Symptoms")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.show()
    print("Saved feature importance as feature_importance.png")

# =========================
# 13. TEST NEW PATIENTS
# =========================
test_cases = [
    ["fever", "cough", "headache"],
    ["vomiting", "fatigue", "nausea"],
    ["chest_pain", "fatigue"],
]

print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)

for i, symptoms in enumerate(test_cases, start=1):
    sample = {col: 0 for col in X.columns}

    valid_symptoms = []
    for s in symptoms:
        if s in sample:
            sample[s] = 1
            valid_symptoms.append(s)

    sample_df = pd.DataFrame([sample])
    pred_label = best_model.predict(sample_df)[0]
    pred_disease = label_encoder.inverse_transform([pred_label])[0]

    print(f"Test Case {i}:")
    print("Symptoms entered:", valid_symptoms)
    print("Predicted Disease:", pred_disease)
    print("-" * 40)