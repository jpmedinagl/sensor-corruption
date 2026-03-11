import os
import glob
import numpy as np

BASE = "../datasets/UCI HAR Dataset/"

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
    files = sorted(glob.glob(os.path.join(path, "*.txt")))

    signals = []
    for f in files:
        data = np.loadtxt(f)
        signals.append(data)

    signals = np.stack(signals, axis=-1)
    return signals


def load_labels():
    y_train = np.loadtxt(os.path.join(BASE, "train/y_train.txt")).astype(int)
    y_test = np.loadtxt(os.path.join(BASE, "test/y_test.txt")).astype(int)
    
    return y_train, y_test


def load_subjects():
    subject_train = np.loadtxt(os.path.join(BASE, "train/subject_train.txt")).astype(int)
    subject_test = np.loadtxt(os.path.join(BASE, "test/subject_test.txt")).astype(int)
    
    return subject_train, subject_test


def load_raw_data():
    X_train = load_raw(os.path.join(BASE, "train/Inertial Signals"))
    X_test = load_raw(os.path.join(BASE, "test/Inertial Signals"))

    y_train, y_test = load_labels()
    
    return X_train, y_train, X_test, y_test


def load_processed_data():
    X_train = np.loadtxt(os.path.join(BASE, "train/X_train.txt"))
    X_test = np.loadtxt(os.path.join(BASE, "test/X_test.txt"))

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

    files = sorted(glob.glob("../datasets/UCI HAR Dataset/train/Inertial Signals/*.txt"))
    for i, f in enumerate(files):
        print(i, os.path.basename(f))
