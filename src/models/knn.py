import os
import sys
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from data import load_raw_data, load_processed_data, GYRO, ACCL
from corruption import CorruptionFramework


def flatten_timeseries(X: np.ndarray) -> np.ndarray:
    """
    Classical ML models expect a 2D input:
        (samples, features)

    Raw HAR data is 3D:
        (samples, timesteps, sensors)

    So we flatten each sample into a 1D feature vector.
    """
    return X.reshape(X.shape[0], -1)


def channel_name(channels):
    if channels == GYRO:
        return "GYRO"
    elif channels == ACCL:
        return "ACCL"
    return str(channels)


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


def evaluate_knn_with_corruption(corruption_type: str, channels, severities):
    X_train, y_train, X_test, y_test = load_raw_data()

    # Train on clean raw data
    X_train_flat = flatten_timeseries(X_train)

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_flat, y_train)

    print(f"=== KNN Corruption Results: {corruption_type}, channels={channel_name(channels)} ===")

    for severity in severities:
        framework = CorruptionFramework(
            corruption_type=corruption_type,
            channels=channels,
            severity=severity
        )

        # Corrupt test data only
        X_test_corrupt = framework.corrupt(X_test)
        X_test_corrupt_flat = flatten_timeseries(X_test_corrupt)

        y_pred = model.predict(X_test_corrupt_flat)

        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro")

        print(f"Severity={severity}: Accuracy={acc:.4f}, Macro-F1={macro_f1:.4f}")

    print()


def main():
    # =========================================================
    # 1. BASELINES
    # =========================================================

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

    # Processed feature baseline
    X_train_proc, y_train_proc, X_test_proc, y_test_proc = load_processed_data()

    evaluate_knn(
        X_train_proc,
        y_train_proc,
        X_test_proc,
        y_test_proc,
        label="processed features"
    )

    # =========================================================
    # 2. MAIN CORRUPTION EXPERIMENTS
    # Focus on the corruption types that produce meaningful
    # degradation patterns and best match the paper direction.
    # =========================================================

    # -------------------------
    # DROPOUT
    # -------------------------
    evaluate_knn_with_corruption(
        corruption_type="dropout",
        channels=GYRO,
        severities=[0.1, 0.2, 0.3, 0.5]
    )

    evaluate_knn_with_corruption(
        corruption_type="dropout",
        channels=ACCL,
        severities=[0.1, 0.2, 0.3, 0.5]
    )

    # -------------------------
    # DRIFT
    # Updated severity range based on discussion:
    # severity controls final accumulated drift in units of std
    # -------------------------
    evaluate_knn_with_corruption(
        corruption_type="drift",
        channels=GYRO,
        severities=[1.0, 2.0, 4.0, 8.0]
    )

    evaluate_knn_with_corruption(
        corruption_type="drift",
        channels=ACCL,
        severities=[1.0, 2.0, 4.0, 8.0]
    )

    # -------------------------
    # BIAS
    # -------------------------
    evaluate_knn_with_corruption(
        corruption_type="bias",
        channels=GYRO,
        severities=[0.5, 1.0, 1.5, 2.0]
    )

    evaluate_knn_with_corruption(
        corruption_type="bias",
        channels=ACCL,
        severities=[0.5, 1.0, 1.5, 2.0]
    )

    # -------------------------
    # GAIN (amplification)
    # gain > 1 may improve performance by amplifying informative
    # channels, so we treat it as a secondary / exploratory result
    # rather than a main corruption result.
    # -------------------------
    evaluate_knn_with_corruption(
        corruption_type="gain",
        channels=GYRO,
        severities=[0.25, 0.5, 0.75, 0.9]
    )

    evaluate_knn_with_corruption(
        corruption_type="gain",
        channels=ACCL,
        severities=[0.25, 0.5, 0.75, 0.9]
    )

    # =========================================================
    # 3. SECONDARY / EXPLORATORY OBSERVATIONS
    # These are interesting, but not the main paper focus.
    # =========================================================

    # -------------------------
    # GAIN (amplification)
    # This may act more like feature reweighting than degradation.
    # -------------------------
    evaluate_knn_with_corruption(
        corruption_type="gain",
        channels=GYRO,
        severities=[1.1, 1.25, 1.5, 2.0]
    )

    evaluate_knn_with_corruption(
        corruption_type="gain",
        channels=ACCL,
        severities=[1.1, 1.25, 1.5, 2.0]
    )

    # -------------------------
    # STOCHASTIC NOISE
    # Keep as an exploratory result for completeness.
    # -------------------------
    evaluate_knn_with_corruption(
        corruption_type="stochastic",
        channels=GYRO,
        severities=[0.25, 0.5, 1.0, 1.25]
    )

    evaluate_knn_with_corruption(
        corruption_type="stochastic",
        channels=ACCL,
        severities=[0.25, 0.5, 1.0, 1.25]
    )

    # -------------------------
    # RESOLUTION
    # Keep as exploratory for now.
    # -------------------------
    evaluate_knn_with_corruption(
        corruption_type="resolution",
        channels=GYRO,
        severities=[1, 2, 3, 4]
    )

    evaluate_knn_with_corruption(
        corruption_type="resolution",
        channels=ACCL,
        severities=[1, 2, 3, 4]
    )


if __name__ == "__main__":
    main()