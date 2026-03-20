import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from data import load_raw_data
from corruption import CorruptionFramework

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.int64)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

class HARLSTM(nn.Module):
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

def train(
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
def test(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        batch_size = y_batch.size(0)
        running_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += batch_size

    return running_loss / total, correct / total

def main():
    batch_size = 64
    epochs = 300

    X_train_val, y_train_val, X_test, y_test = load_raw_data()
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, stratify=y_train_val)
    print(X_train.shape)
    print(X_val.shape)
    print(X_test.shape)

    train_dataloader = make_loader(X_train, y_train, batch_size, True)
    val_dataloader = make_loader(X_val, y_val, batch_size, False)
    test_dataloader = make_loader(X_test, y_test, batch_size, False)
    
    model = HARLSTM(
        input_size=X_train.shape[-1],
        hidden_size=128,
        dense_size=64,
        num_classes=6,
        dropout=0.2,
    ).to(device)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train(model, train_dataloader, loss_fn, optimizer, device)
        val_loss, val_acc = test(model, val_dataloader, loss_fn, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:03d}/{epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc = test(model, test_dataloader, loss_fn, device)
    print(f"Test | loss={test_loss:.4f} acc={test_acc:.4f}")

    torch.save(model.state_dict(), "lstm.pth")
    print("Saved PyTorch Model State to lstm.pth")

if __name__ == "__main__":
    main()
