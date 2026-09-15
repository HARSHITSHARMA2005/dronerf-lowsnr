"""Conclusive shuffled-label test: run long enough to see the plateau, with
naive baselines and a real-label reference for context.

If X leaks the true label, a model trained on permuted labels still predicts
the true val label => val_acc stays high and stable. If not, val_acc settles
around the class-prior baselines (30-37%) with epoch-to-epoch noise.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
D = REPO_ROOT / "data" / "processed"
SEED = 42


class MLP(nn.Module):
    def __init__(self, in_dim=2048, n=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, n))

    def forward(self, x):
        return self.net(x)


def run(tag, Xtr, ytr, Xva, yva, epochs=20):
    torch.manual_seed(SEED)
    model = MLP()
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    crit = nn.CrossEntropyLoss()
    n = Xtr.shape[0]
    accs = []
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 128):
            b = perm[i:i + 128]
            opt.zero_grad()
            crit(model(Xtr[b]), ytr[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            tr = (model(Xtr).argmax(1) == ytr).float().mean().item()
            va = (model(Xva).argmax(1) == yva).float().mean().item()
        accs.append(va)
        if ep % 4 == 0 or ep <= 3:
            print(f"  [{tag}] epoch {ep:2d}: train_acc={tr*100:5.2f}%  val_acc={va*100:5.2f}%")
    accs = np.array(accs[-10:])
    print(f"  [{tag}] last-10-epoch val_acc: mean={accs.mean()*100:.2f}%  "
          f"min={accs.min()*100:.2f}%  max={accs.max()*100:.2f}%\n")


def main():
    Xtr = torch.from_numpy(np.load(D / "X_train.npy")).float()
    Xva = torch.from_numpy(np.load(D / "X_val.npy")).float()
    ytr = np.load(D / "y_train.npy")
    yva = np.load(D / "y_val.npy")

    freqs = np.bincount(yva) / len(yva)
    print(f"val class freqs: {np.round(freqs, 3).tolist()}")
    print(f"naive baselines -> predict-mode: {freqs.max()*100:.1f}%   "
          f"proportional-random: {(freqs**2).sum()*100:.1f}%\n")

    yva_t = torch.from_numpy(yva).long()

    # 1) real labels - reference
    run("REAL   ", Xtr, torch.from_numpy(ytr).long(), Xva, yva_t)

    # 2) train labels permuted, val untouched
    rng = np.random.RandomState(SEED)
    ytr_s = ytr.copy(); rng.shuffle(ytr_s)
    run("SHUF-TR", Xtr, torch.from_numpy(ytr_s).long(), Xva, yva_t)

    # 3) both train and val permuted with independent perms (sanity: same as #2)
    yva_s = yva.copy(); rng.shuffle(yva_s)
    run("SHUF-BOTH", Xtr, torch.from_numpy(ytr_s).long(), Xva, torch.from_numpy(yva_s).long())


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"(total {time.time()-t0:.0f}s)")
