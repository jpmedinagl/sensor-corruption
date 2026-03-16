import os
import sys
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from data import load_raw_data, load_processed_data

def flatten_timeseries(X: np.ndarray) -> np.ndarray:
    '''The expected input is a two-dimensional matrix, 
    where each row represents a sample and each column represents a feature. 
    The original data is three-dimensional (samples * timesteps * sensors) 
    and cannot be directly input into these models. 
    Therefore, it is necessary to "flatten" the values of all time points 
    and all channels of each sample into a one-dimensional feature vector.
    '''
    return X.reshape(X.shape[0], -1)

def evaluate_knn(X_train, y_train, X_test, y_test, label: str, n_neighbors: int = 5):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)

    print(f"=== KNN Results ({label}) ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print()

def main():
    # Raw data baseline
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = load_raw_data()
    X_train_raw = flatten_timeseries(X_train_raw)
    X_test_raw = flatten_timeseries(X_test_raw)

    evaluate_knn(
        X_train_raw,
        y_train_raw,
        X_test_raw,
        y_test_raw,
        label="raw flattened"
    )

    # Processed data baseline
    X_train_proc, y_train_proc, X_test_proc, y_test_proc = load_processed_data()

    evaluate_knn(
        X_train_proc,
        y_train_proc,
        X_test_proc,
        y_test_proc,
        label="processed features"
    )


if __name__ == "__main__":
    main()