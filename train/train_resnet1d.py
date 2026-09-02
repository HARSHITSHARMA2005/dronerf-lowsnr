"""1D ResNet baseline for DroneRF 4-class classification (PyTorch, CPU)."""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_splits, save_results, RESULTS_DIR

RANDOM_STATE = 42
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
PATIENCE = 5
NUM_CLASSES = 4
MAX_TRAIN_SECONDS = 1800  # abort guard: 30 minutes

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        identity = x if self.shortcut is None else self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return self.relu(out)


class ResNet1D(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.block1 = ResidualBlock1D(32, 32, kernel_size=5)
        self.block2 = ResidualBlock1D(32, 64, kernel_size=5, stride=2)
        self.block3 = ResidualBlock1D(64, 128, kernel_size=5, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


def make_loader(X, y, batch_size, shuffle):
    X_t = torch.from_numpy(X).float().unsqueeze(1)  # (N, 1, 2048)
    y_t = torch.from_numpy(y).long()
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
    return total_loss / total, correct / total


def main():
    device = torch.device("cpu")
    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()
    print(f"Loaded splits: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

    train_loader = make_loader(X_train, y_train, BATCH_SIZE, shuffle=True)
    val_loader = make_loader(X_val, y_val, BATCH_SIZE, shuffle=False)
    test_loader = make_loader(X_test, y_test, BATCH_SIZE, shuffle=False)

    model = ResNet1D().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        n_samples = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            n_samples += xb.size(0)
        train_loss = running_loss / n_samples

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch:02d}/{EPOCHS} | train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | val_acc={val_acc * 100:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (no val improvement for {PATIENCE} epochs).")
                break

        if time.time() - t0 > MAX_TRAIN_SECONDS:
            print(f"ABORT: training exceeded {MAX_TRAIN_SECONDS}s. Stopping early.")
            break

    train_time = time.time() - t0

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    all_logits = []
    all_preds = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            all_logits.append(logits.numpy())
            all_preds.append(logits.argmax(dim=1).numpy())
    test_logits = np.concatenate(all_logits, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)

    out_dir = RESULTS_DIR / "baseline_resnet1d"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")

    save_results("resnet1d", y_test, y_pred, test_logits, train_time)


if __name__ == "__main__":
    main()
