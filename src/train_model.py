"""
Model Training Module for AI-Based Intrusion Detection System
Trains and compares multiple ML models for best performance
"""

import numpy as np
import pandas as pd
import joblib
import os
import json
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score

from src.preprocess import IDSPreprocessor


class IDSModelTrainer:
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, max_depth=20, random_state=42, n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=1000, random_state=42
            ),
            'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        }
        self.results = {}
        self.best_model = None
        self.best_model_name = None

    def train_all(self, X_train, y_train):
        """Train all models."""
        print("\n" + "="*60)
        print("  TRAINING MODELS")
        print("="*60)

        for name, model in self.models.items():
            print(f"\n[*] Training {name}...")
            start = datetime.now()
            model.fit(X_train, y_train)
            elapsed = (datetime.now() - start).seconds
            print(f"[+] {name} trained in {elapsed}s")

    def evaluate_all(self, X_test, y_test):
        """Evaluate all models and pick the best."""
        print("\n" + "="*60)
        print("  EVALUATION RESULTS")
        print("="*60)

        best_f1 = 0

        for name, model in self.models.items():
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            cm = confusion_matrix(y_test, y_pred).tolist()

            self.results[name] = {
                'accuracy': round(acc, 4),
                'precision': round(prec, 4),
                'recall': round(rec, 4),
                'f1_score': round(f1, 4),
                'confusion_matrix': cm
            }

            print(f"\n{'─'*40}")
            print(f"  Model     : {name}")
            print(f"  Accuracy  : {acc*100:.2f}%")
            print(f"  Precision : {prec*100:.2f}%")
            print(f"  Recall    : {rec*100:.2f}%")
            print(f"  F1 Score  : {f1*100:.2f}%")

            if f1 > best_f1:
                best_f1 = f1
                self.best_model = model
                self.best_model_name = name

        print(f"\n{'='*60}")
        print(f"  BEST MODEL: {self.best_model_name}")
        print(f"  BEST F1   : {best_f1*100:.2f}%")
        print("="*60)

    def save_best_model(self, path='models/best_model.pkl'):
        """Save the best performing model."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.best_model, path)
        print(f"\n[+] Best model saved: {path}")

    def save_results(self, path='models/results.json'):
        """Save evaluation results to JSON."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump({
                'best_model': self.best_model_name,
                'results': self.results,
                'timestamp': datetime.now().isoformat()
            }, f, indent=4)
        print(f"[+] Results saved: {path}")

    def print_classification_report(self, X_test, y_test):
        """Print detailed classification report for best model."""
        print(f"\n[CLASSIFICATION REPORT - {self.best_model_name}]")
        y_pred = self.best_model.predict(X_test)
        print(classification_report(y_test, y_pred,
              target_names=['Normal', 'Attack']))


def main():
    # Step 1: Preprocess data
    preprocessor = IDSPreprocessor()
    X_train, X_test, y_train, y_test, df = preprocessor.preprocess(
        r'C:\Users\Chethan\Desktop\AI_IDS_Project\data\raw\KDDTrain+.txt'
    )
    preprocessor.save(r'C:\Users\Chethan\Desktop\AI_IDS_Project\models\preprocessor.pkl')

    # Step 2: Train models
    trainer = IDSModelTrainer()
    trainer.train_all(X_train, y_train)

    # Step 3: Evaluate models
    trainer.evaluate_all(X_test, y_test)
    trainer.print_classification_report(X_test, y_test)

    # Step 4: Save best model and results
    trainer.save_best_model(r'C:\Users\Chethan\Desktop\AI_IDS_Project\models\best_model.pkl')
    trainer.save_results(r'C:\Users\Chethan\Desktop\AI_IDS_Project\models\results.json')

    print("\n[+] Training pipeline complete!")


if __name__ == "__main__":
    main()