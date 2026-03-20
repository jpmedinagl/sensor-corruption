from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent.parent / "datasets" / "UCI HAR Dataset"

CHANNELS = {
    "body_acc_x": 0,
    "body_acc_y": 1,
    "body_acc_z": 2,
    "body_gyro_x": 3,
    "body_gyro_y": 4,
    "body_gyro_z": 5,
    "total_acc_x": 6,
    "total_acc_y": 7,
    "total_acc_z": 8,
}

ACCL = [0, 1, 2]
GYRO = [3, 4, 5]


def load_raw(path):
    files = sorted(Path(path).glob("*.txt"))

    signals = []
    for f in files:
        data = np.loadtxt(f)
        signals.append(data)

    signals = np.stack(signals, axis=-1)
    return signals


def load_labels():
    y_train = np.loadtxt(BASE / "train" / "y_train.txt").astype(int) - 1
    y_test = np.loadtxt(BASE / "test" / "y_test.txt").astype(int) - 1
    
    return y_train, y_test


def load_subjects():
    subject_train = np.loadtxt(BASE / "train" / "subject_train.txt").astype(int)
    subject_test = np.loadtxt(BASE / "test" / "subject_test.txt").astype(int)
    
    return subject_train, subject_test


def load_raw_data():
    X_train = load_raw(BASE / "train" / "Inertial Signals")
    X_test = load_raw(BASE / "test" / "Inertial Signals")

    y_train, y_test = load_labels()
    
    return X_train, y_train, X_test, y_test


def load_processed_data():
    X_train = np.loadtxt(BASE / "train" / "X_train.txt")
    X_test = np.loadtxt(BASE / "test" / "X_test.txt")

    y_train, y_test = load_labels()

    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_raw_data()
    
    print("(samples, timesteps, sensors)")
    print(f"X_train.shape {X_train.shape}")
    print(f"y_train.shape {y_train.shape}")
    print(f"X_test.shape {X_test.shape}")
    print(f"y_test.shape {y_test.shape}")

    X_train, y_train, X_test, y_test = load_processed_data()
    
    print("(samples, features)")
    print(f"X_train.shape {X_train.shape}")
    print(f"y_train.shape {y_train.shape}")
    print(f"X_test.shape {X_test.shape}")
    print(f"y_test.shape {y_test.shape}")

    files = sorted((BASE / "train" / "Inertial Signals").glob("*.txt"))
    for i, f in enumerate(files):
        print(i, f.name)
