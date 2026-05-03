"""
Data Preprocessing Module for AI-Based Intrusion Detection System
NSL-KDD Dataset
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os
import joblib

# NSL-KDD column names
COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
    'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
    'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

# Attack category mapping
ATTACK_CATEGORY = {
    'normal': 'normal',
    'neptune': 'DoS', 'back': 'DoS', 'land': 'DoS', 'pod': 'DoS',
    'smurf': 'DoS', 'teardrop': 'DoS', 'mailbomb': 'DoS',
    'apache2': 'DoS', 'processtable': 'DoS', 'udpstorm': 'DoS',
    'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe',
    'satan': 'Probe', 'mscan': 'Probe', 'saint': 'Probe',
    'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L',
    'multihop': 'R2L', 'phf': 'R2L', 'spy': 'R2L', 'warezclient': 'R2L',
    'warezmaster': 'R2L', 'sendmail': 'R2L', 'named': 'R2L',
    'snmpgetattack': 'R2L', 'snmpguess': 'R2L', 'xlock': 'R2L',
    'xsnoop': 'R2L', 'worm': 'R2L',
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R',
    'rootkit': 'U2R', 'httptunnel': 'U2R', 'ps': 'U2R',
    'sqlattack': 'U2R', 'xterm': 'U2R'
}

class IDSPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = None

    def load_data(self, filepath):
        """Load NSL-KDD dataset from file."""
        print(f"[*] Loading data from: {filepath}")
        df = pd.read_csv(filepath, names=COLUMNS, header=None)
        print(f"[+] Loaded {len(df)} records with {len(df.columns)} features")
        return df

    def map_attack_categories(self, df):
        """Map specific attack names to attack categories."""
        df = df.copy()
        df['label'] = df['label'].str.strip('.')
        df['attack_category'] = df['label'].map(
            lambda x: ATTACK_CATEGORY.get(x.lower(), 'Unknown')
        )
        df['binary_label'] = df['label'].apply(
            lambda x: 0 if x.lower() == 'normal' else 1
        )
        return df

    def encode_categorical(self, df, fit=True):
        """Encode categorical features."""
        df = df.copy()
        categorical_cols = ['protocol_type', 'service', 'flag']

        for col in categorical_cols:
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]
                df[col] = df[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
        return df

    def get_features(self, df):
        """Extract feature columns (drop label columns)."""
        drop_cols = ['label', 'difficulty', 'attack_category', 'binary_label']
        drop_cols = [c for c in drop_cols if c in df.columns]
        features = df.drop(columns=drop_cols)
        self.feature_columns = features.columns.tolist()
        return features

    def scale_features(self, X, fit=True):
        """Normalize numerical features."""
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        return X_scaled

    def preprocess(self, filepath, test_size=0.2):
        """Full preprocessing pipeline."""
        df = self.load_data(filepath)
        df = self.map_attack_categories(df)
        df = self.encode_categorical(df, fit=True)

        X = self.get_features(df)
        y_binary = df['binary_label']
        y_multi = df['attack_category']

        X_scaled = self.scale_features(X, fit=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_binary, test_size=test_size, random_state=42, stratify=y_binary
        )

        print(f"[+] Training samples: {len(X_train)}")
        print(f"[+] Testing samples:  {len(X_test)}")
        print(f"[+] Features:         {X_scaled.shape[1]}")
        print(f"[+] Attack types: {df['attack_category'].value_counts().to_dict()}")

        return X_train, X_test, y_train, y_test, df

    def save(self, path='models/preprocessor.pkl'):
        """Save preprocessor for later use."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"[+] Preprocessor saved to {path}")

    @staticmethod
    def load(path='models/preprocessor.pkl'):
        """Load a saved preprocessor."""
        return joblib.load(path)


if __name__ == "__main__":
    preprocessor = IDSPreprocessor()
    X_train, X_test, y_train, y_test, df = preprocessor.preprocess(
        'data/raw/KDDTrain+.txt'
    )
    preprocessor.save('models/preprocessor.pkl')
    print("[+] Preprocessing complete!")