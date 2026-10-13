import os
import glob
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from nn_models import MotionNet

files = glob.glob("data/sequences/*.npy")
if not files:
    raise SystemExit("No sequences found. Run collect_sequences.py first.")

X, names = [], []
for f in files:
    X.append(np.load(f))
    names.append(os.path.basename(f).rsplit("_", 1)[0])
X = np.array(X, dtype=np.float32)
classes = sorted(set(names))
if len(classes) < 2:
    raise SystemExit("Record at least 2 different motion signs.")
y = np.array([classes.index(n) for n in names])
seq_len = X.shape[1]
print(f"{len(X)} sequences, classes: {classes}")

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
Xtr, Xte = torch.tensor(Xtr), torch.tensor(Xte)
ytr, yte = torch.tensor(ytr), torch.tensor(yte)


def run(cell, bidirectional):
    torch.manual_seed(42)
    model = MotionNet(len(classes), cell, bidirectional)
    opt = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(80):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 16):
            idx = perm[i:i + 16]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # guards against exploding gradients
            opt.step()
    model.eval()
    with torch.no_grad():
        acc = (model(Xte).argmax(1) == yte).float().mean().item()
    return model, acc


results = {}
for name, cell, bi in [("LSTM", "lstm", False), ("GRU", "gru", False), ("BiLSTM", "lstm", True)]:
    model, acc = run(cell, bi)
    results[name] = (model, acc, cell, bi)
    print(f"{name:7s} test accuracy: {acc:.3f}")

best = max(results, key=lambda k: results[k][1])
model, acc, cell, bi = results[best]
os.makedirs("models", exist_ok=True)
torch.save({"state_dict": model.state_dict(), "classes": classes,
            "cell": cell, "bidirectional": bi, "seq_len": seq_len}, "models/motion_model.pt")
print(f"\nBest model: {best} ({acc:.3f}) saved to models/motion_model.pt")