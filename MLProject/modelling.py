import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import os

mlflow.sklearn.autolog()

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Credit_Card_Fraud_Detection")
os.environ["LOGNAME"] = "ivan"
os.environ["USER"] = "ivan"
os.environ["USERNAME"] = "ivan"

def load_data():
    dataset_path = "namadataset_preprocessing/creditcard_preprocessed.csv"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}.")
        
    df = pd.read_csv(dataset_path)
    X = df.drop("Class", axis=1)
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test

def train_model():
    X_train, X_test, y_train, y_test = load_data()

    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    mlflow.sklearn.log_model(model, "model")

    print("Training completed and logged locally via autolog.")

if __name__ == "__main__":
    train_model()