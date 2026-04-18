import mlflow
import mlflow.sklearn
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import pandas as pd
import numpy as np
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

dagshub.init(repo_owner='ivan8419', repo_name='SMSML_Muhammad_Ivan', mlflow=True)

mlflow.set_experiment("Credit_Card_Fraud_Detection_Tuning")
os.environ["LOGNAME"] = "ivan"
os.environ["USER"] = "ivan"
os.environ["USERNAME"] = "ivan"

def load_data():
    dataset_path = "namadataset_preprocessing/creditcard_preprocessed.csv"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset {dataset_path} not found. Run preprocessing first.")
    df = pd.read_csv(dataset_path)
    X = df.drop("Class", axis=1)
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test

def train_tuning():
    X_train, X_test, y_train, y_test = load_data()

    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced', 'balanced_subsample']
    }

    rf = RandomForestClassifier(random_state=42)
    random_search = RandomizedSearchCV(
        rf, param_distributions=param_dist, n_iter=10, cv=3,
        scoring='f1', random_state=42, n_jobs=-1
    )

    with mlflow.start_run():
        random_search.fit(X_train, y_train)

        best_model = random_search.best_estimator_
        preds = best_model.predict(X_test)
        pred_proba = best_model.predict_proba(X_test)[:, 1]

        mlflow.log_params(random_search.best_params_)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        roc_auc = roc_auc_score(y_test, pred_proba)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        cm = confusion_matrix(y_test, preds)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        plt.tight_layout()
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path)

        feature_importance = best_model.feature_importances_
        fi_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)

        fig2, ax2 = plt.subplots(figsize=(10, 8))
        ax2.barh(range(len(fi_df)), fi_df['importance'])
        ax2.set_yticks(range(len(fi_df)))
        ax2.set_yticklabels(fi_df['feature'])
        ax2.set_title('Feature Importance')
        ax2.invert_yaxis()
        plt.tight_layout()
        fi_path = "feature_importance.png"
        fig2.savefig(fi_path)
        plt.close()
        mlflow.log_artifact(fi_path)

        mlflow.sklearn.log_model(best_model, "model")

        print("Model Tuning completed. Logged to DagsHub via MLflow.")
        print(f"Best params: {random_search.best_params_}")
        print(f"Metrics - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")

if __name__ == "__main__":
    train_tuning()