import pandas as pd
import mlflow
import shutil
import os
import joblib
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load Data
data = pd.read_csv("dataset_preprocess/telco_preprocessed.csv")

X_train, X_test, y_train, y_test = train_test_split(
    data.drop("Churn", axis=1),
    data["Churn"],
    random_state=42,
    test_size=0.2,
    stratify=data["Churn"]
)

mlflow.sklearn.autolog()

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print(f"Accuracy : {acc:.4f}")

mlflow.log_metric("accuracy", acc)

os.makedirs("saved_model", exist_ok=True)

# Simpan model
joblib.dump(model, "saved_model/model.pkl")

# Simpan model MLflow
model_path = "saved_model/mlflow_model"

if os.path.exists(model_path):
    shutil.rmtree(model_path)

mlflow.sklearn.save_model(
    sk_model=model,
    path=model_path
)

print("Training selesai.")