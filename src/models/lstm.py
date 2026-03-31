import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from data import load_raw_data, GYRO, ACCL
from corruption import CorruptionFramework


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.int64)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

class HAR_LSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dense_size: int,
        num_classes: int,
        dropout: float,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, dense_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dense_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]
        h_drop = self.dropout(h_last)
        h_dense = self.relu(self.fc1(h_drop))
        logits = self.fc2(h_dense)
        return logits

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        batch_size = y_batch.size(0)
        running_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += batch_size

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    y_true_all = []
    y_pred_all = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        preds = logits.argmax(dim=1)
        batch_size = y_batch.size(0)
        running_loss += loss.item() * batch_size
        correct += (preds == y_batch).sum().item()
        total += batch_size

        y_true_all.append(y_batch.detach().cpu().numpy())
        y_pred_all.append(preds.detach().cpu().numpy())

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    return running_loss / total, correct / total, y_true, y_pred

def evaluate_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label: str,
    epochs: int = 80,
    batch_size: int = 64,
) -> HAR_LSTM:

    X_subtrain, X_val, y_subtrain, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=42,
    )

    train_loader = make_loader(X_subtrain, y_subtrain, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, batch_size=batch_size, shuffle=False)
    test_loader = make_loader(X_test, y_test, batch_size=batch_size, shuffle=False)

    model = HAR_LSTM(
        input_size=X_train.shape[-1],
        hidden_size=128,
        dense_size=64,
        num_classes=6,
        dropout=0.2,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate_epoch(model, val_loader, criterion, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:03d}/{epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc, y_true, y_pred = evaluate_epoch(model, test_loader, criterion, device)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)

    print(f"=== LSTM Results ({label}) ===")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print()

    return model


def evaluate_lstm_with_corruption(
    corruption_type: str,
    channels,
    severities,
    corrupted_dataset: str = "test",
    epochs: int = 80,
    batch_size: int = 64,
):
    X_train, y_train, X_test, y_test = load_raw_data()

    if corrupted_dataset not in {"train", "test", "both"}:
        raise ValueError("corrupted_dataset must be one of: train, test, both")

    print(
        f"=== LSTM Corruption Results: {corruption_type}, "
        f"channels={channels}, dataset={corrupted_dataset} ==="
    )

    criterion = nn.CrossEntropyLoss()

    model = None
    if corrupted_dataset == "test":
        model = evaluate_lstm(
            X_train,
            y_train,
            X_test,
            y_test,
            label=f"clean train baseline for {corruption_type}",
            epochs=epochs,
            batch_size=batch_size,
        )

    for severity in severities:
        framework = CorruptionFramework(
            corruption_type=corruption_type,
            channels=channels,
            severity=severity,
        )

        X_train_eval = X_train
        X_test_eval = X_test

        if corrupted_dataset in {"train", "both"}:
            X_train_eval = framework.corrupt(X_train)

        if corrupted_dataset in {"test", "both"}:
            X_test_eval = framework.corrupt(X_test)

        if corrupted_dataset in {"train", "both"}:
            model = evaluate_lstm(
                X_train_eval,
                y_train,
                X_test_eval,
                y_test,
                label=(
                    f"{corruption_type} severity={severity} "
                    f"(dataset={corrupted_dataset})"
                ),
                epochs=epochs,
                batch_size=batch_size,
            )
            continue

        test_loader = make_loader(X_test_eval, y_test, batch_size=batch_size, shuffle=False)
        _, _, y_true, y_pred = evaluate_epoch(model, test_loader, criterion, device)

        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro")

        print(f"Severity={severity}: Accuracy={acc:.4f}, Macro-F1={macro_f1:.4f}")

    print()

def sweep():
    channels = [GYRO, ACCL]
    corruption = {"dropout": [0.1, 0.3, 0.5],
                  "drift": [0.5, 1.0, 2.0, 4.0],
                  "stochastic": [0.25, 0.5, 1.0, 1.25],
                  "bias": [0.5, 1.0, 1.25, 2.0],
                  "gain": [0.5, 0.75, 1.25, 2.0],
                  "resolution": [1, 2, 3, 4]
                }
    corrupted_datasets = ['train', 'test', 'both']

    for corrupted_dataset in corrupted_datasets:
        print(f"\n===== Corrupt dataset mode: {corrupted_dataset} =====")

        for corruption_type, severities in corruption.items():
            for channel in channels:
                evaluate_lstm_with_corruption(
                    corruption_type=corruption_type,
                    channels=channel,
                    severities=severities,
                    corrupted_dataset=corrupted_dataset,
                )

    print()

def main():
    # Raw data baseline
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = load_raw_data()

    baseline_model = evaluate_lstm(
        X_train_raw,
        y_train_raw,
        X_test_raw,
        y_test_raw,
        label="raw sequence",
    )

    torch.save(baseline_model.state_dict(), "lstm.pth")
    print("Saved PyTorch model state to lstm.pth")
    print()

    # # Gyroscope dropout
    # evaluate_lstm_with_corruption(
    #     corruption_type="dropout",
    #     channels=GYRO,
    #     severities=[0.1, 0.3, 0.5]
    # )

    # # Accelerometer dropout
    # evaluate_lstm_with_corruption(
    #     corruption_type="dropout",
    #     channels=ACCL,
    #     severities=[0.1, 0.3, 0.5]
    # )

    # # Gyroscope drift
    # evaluate_lstm_with_corruption(
    #     corruption_type="drift",
    #     channels=GYRO,
    #     severities=[0.5, 1.0, 2.0, 4.0]
    # )

    # # Accelerometer drift
    # evaluate_lstm_with_corruption(
    #     corruption_type="drift",
    #     channels=ACCL,
    #     severities=[0.5, 1.0, 2.0, 4.0]
    # )

    # # Gyroscope stochastic noise
    # evaluate_lstm_with_corruption(
    #     corruption_type="stochastic",
    #     channels=GYRO,
    #     severities=[0.25, 0.5, 1.0, 1.25]
    # )

    # # Accelerometer stochastic noise
    # evaluate_lstm_with_corruption(
    #     corruption_type="stochastic",
    #     channels=ACCL,
    #     severities=[0.25, 0.5, 1.0, 1.25]
    # )

    # # Gyroscope bias
    # evaluate_lstm_with_corruption(
    #     corruption_type="bias",
    #     channels=GYRO,
    #     severities=[0.5, 1.0, 1.25, 2.0]
    # )

    # # Accelerometer bias
    # evaluate_lstm_with_corruption(
    #     corruption_type="bias",
    #     channels=ACCL,
    #     severities=[0.5, 1.0, 1.25, 2.0]
    # )

    # # Gyroscope gain
    # evaluate_lstm_with_corruption(
    #     corruption_type="gain",
    #     channels=GYRO,
    #     severities=[0.5, 0.75, 1.25, 2.0]
    # )

    # # Accelerometer gain
    # evaluate_lstm_with_corruption(
    #     corruption_type="gain",
    #     channels=ACCL,
    #     severities=[0.5, 0.75, 1.25, 2.0]
    # )

    # # Gyroscope resolution
    # evaluate_lstm_with_corruption(
    #     corruption_type="resolution",
    #     channels=GYRO,
    #     severities=[1, 2, 3, 4]
    # )

    # # Accelerometer resolution
    # evaluate_lstm_with_corruption(
    #     corruption_type="resolution",
    #     channels=ACCL,
    #     severities=[1, 2, 3, 4]
    # )

    # Full sweep across train/test/both corruption modes.
    sweep()

if __name__ == "__main__":
    main()
