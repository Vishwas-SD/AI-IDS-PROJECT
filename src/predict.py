"""
Prediction Module for AI-Based Intrusion Detection System
Use this to classify network traffic as Normal or Attack
"""

import numpy as np
import pandas as pd
import joblib
import json
import sys
import os
sys.path.append(r'C:\Users\Chethan\Desktop\AI_IDS_PROJECT\src')
from preprocess import IDSPreprocessor, COLUMNS, ATTACK_CATEGORY

class IDSPredictor:
    def __init__(self, model_path=r'C:\Users\Chethan\Desktop\AI_IDS_PROJECT\models\best_model.pkl',
                 preprocessor_path=r'C:\Users\Chethan\Desktop\AI_IDS_PROJECT\models\preprocessor.pkl'):
        print("[*] Loading IDS model...")
        self.model = joblib.load(model_path)
        self.preprocessor = IDSPreprocessor.load(preprocessor_path)
        print("[+] Model loaded and ready!")

    def predict_from_file(self, filepath):
        """Predict on a full test file."""
        df = self.preprocessor.load_data(filepath)
        df = self.preprocessor.map_attack_categories(df)
        df = self.preprocessor.encode_categorical(df, fit=False)

        X = self.preprocessor.get_features(df)
        X_scaled = self.preprocessor.scale_features(X, fit=False)

        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]

        df['prediction'] = predictions
        df['attack_probability'] = probabilities
        df['prediction_label'] = df['prediction'].map({0: 'Normal', 1: 'Attack'})

        return df

    def predict_single(self, traffic_dict):
        """
        Predict on a single network traffic record.
        traffic_dict: dict with feature values
        """
        df = pd.DataFrame([traffic_dict], columns=[
            c for c in COLUMNS if c not in ['label', 'difficulty']
        ])
        df = self.preprocessor.encode_categorical(df, fit=False)
        X_scaled = self.preprocessor.scale_features(df, fit=False)

        prediction = self.model.predict(X_scaled)[0]
        probability = self.model.predict_proba(X_scaled)[0][1]

        return {
    'prediction': 'Attack' if prediction == 1 else 'Normal',
    'attack_probability': round(float(probability), 4),
    'confidence': f"{max(probability, 1-probability)*100:.1f}%",
    'alert': bool(prediction == 1)
}
        

    def evaluate_on_test(self, test_filepath):
        """Evaluate model on test file and print metrics."""
        from sklearn.metrics import classification_report, accuracy_score

        df = self.predict_from_file(test_filepath)
        y_true = df['binary_label']
        y_pred = df['prediction']

        print("\n" + "="*60)
        print("  TEST SET EVALUATION")
        print("="*60)
        print(f"  Accuracy: {accuracy_score(y_true, y_pred)*100:.2f}%")
        print(classification_report(y_true, y_pred,
              target_names=['Normal', 'Attack']))

        # Attack distribution
        print("Attack Distribution in Predictions:")
        print(df['prediction_label'].value_counts())

        return df


def main():
    predictor = IDSPredictor()

    # Evaluate on test set
    print("\n[*] Evaluating on NSL-KDD test set...")
    results = predictor.evaluate_on_test(r'C:\Users\Chethan\Desktop\AI_IDS_Project\data\raw\KDDTest+.txt')

    # Show sample predictions
    print("\n[SAMPLE PREDICTIONS]")
    sample = results[['protocol_type', 'service', 'flag',
                       'prediction_label', 'attack_probability']].head(10)
    print(sample.to_string(index=False))


if __name__ == "__main__":
    main()