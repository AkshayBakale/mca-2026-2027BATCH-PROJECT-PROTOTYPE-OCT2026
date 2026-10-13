from features import normalize
import os
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from nn_models import AutoEncoder

DATA_PATH = "data/landmarks/landmark_data.csv"
os.makedirs("models", exist_ok=True)

df = pd.read_csv(DATA_PATH)
X = normalize(df.drop("label", axis=1).values)

scaler = StandardScaler()
Xs = scaler.fit_transform(X).astype(np.float32)
Xtr, Xte = train_test_split(Xs, test_size=0.2, random_state=42)
Xtr, Xte = torch.tensor(Xtr), torch.tensor(Xte)

torch.manual_seed(42)
model = AutoEncoder(Xtr.shape[1])
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(60):
    model.train()
    perm = torch.randperm(len(Xtr))
    for i in range(0, len(Xtr), 64):
        batch = Xtr[perm[i:i + 64]]
        noisy = batch + 0.1 * torch.randn_like(batch)      # denoising: corrupt input, reconstruct clean
        opt.zero_grad()
        loss = loss_fn(model(noisy), batch)
        loss.backward()
        opt.step()
    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(Xte), Xte).item()
        print(f"Epoch {epoch + 1}/60  val loss: {val:.5f}")

model.eval()
with torch.no_grad():
    errors = torch.mean((model(Xte) - Xte) ** 2, dim=1).numpy()
threshold = float(np.percentile(errors, 95) * 1.2)
print(f"Reconstruction error threshold: {threshold:.5f}")

torch.save(model.state_dict(), "models/autoencoder.pt")
joblib.dump(scaler, "models/autoencoder_scaler.pkl")
joblib.dump(threshold, "models/autoencoder_threshold.pkl")
print("Saved autoencoder, scaler and threshold.")