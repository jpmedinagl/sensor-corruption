import os
import sys
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from data import load_raw_data, load_processed_data, GYRO, ACCL
from corruption import CorruptionFramework

def flatten_timeseries(X: np.ndarray) -> np.ndarray:
    '''The expected input is a two-dimensional matrix, 
    where each row represents a sample and each column represents a feature. 
    The original data is three-dimensional (samples * timesteps * sensors) 
    and cannot be directly input into these models. 
    Therefore, it is necessary to "flatten" the values of all time points 
    and all channels of each sample into a one-dimensional feature vector.
    '''
    return X.reshape(X.shape[0], -1)

def build_svm_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LinearSVC(max_iter=5000))
    ])

def evaluate_svm(X_train, y_train, X_test, y_test, label: str):
    model = build_svm_model()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)

    print(f"=== SVM Results ({label}) ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print()

def evaluate_svm_with_corruption(corruption_type, channels, severities):
    X_train, y_train, X_test, y_test = load_raw_data()

    X_train_flat = flatten_timeseries(X_train)

    model = build_svm_model()
    model.fit(X_train_flat, y_train)

    print(f"\n=== SVM Corruption Results: {corruption_type}, channels={channels} ===")

    for severity in severities:
        framework = CorruptionFramework(
            corruption_type=corruption_type,
            channels=channels,
            severity=severity
        )

        X_test_corrupt = framework.corrupt(X_test)
        X_test_flat = flatten_timeseries(X_test_corrupt)

        y_pred = model.predict(X_test_flat)

        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro")

        print(f"Severity={severity}: Accuracy={acc:.4f}, Macro-F1={macro_f1:.4f}")

def main():
    # Raw data baseline
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = load_raw_data()
    X_train_raw = flatten_timeseries(X_train_raw)
    X_test_raw = flatten_timeseries(X_test_raw)

    evaluate_svm(
        X_train_raw,
        y_train_raw,
        X_test_raw,
        y_test_raw,
        label="raw flattened"
    )

    # Processed data baseline
    X_train_proc, y_train_proc, X_test_proc, y_test_proc = load_processed_data()

    evaluate_svm(
        X_train_proc,
        y_train_proc,
        X_test_proc,
        y_test_proc,
        label="processed features"
    )

    # Gyroscope dropout
    evaluate_svm_with_corruption(
        corruption_type="dropout",
        channels=GYRO,
        severities=[0.1, 0.3, 0.5]
    )

    # Accelerometer dropout
    evaluate_svm_with_corruption(
        corruption_type="dropout",
        channels=ACCL,
        severities=[0.1, 0.3, 0.5]
    )

    # Gyroscope drift
    evaluate_svm_with_corruption(
        corruption_type="drift",
        channels=GYRO,
        severities=[0.1, 0.3, 0.5]
    )

    # Accelerometer drift
    evaluate_svm_with_corruption(
        corruption_type="drift",
        channels=ACCL,
        severities=[0.1, 0.3, 0.5]
    )
    
if __name__ == "__main__":
    main()