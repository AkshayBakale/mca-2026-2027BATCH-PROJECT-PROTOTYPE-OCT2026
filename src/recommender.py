import numpy as np
import pandas as pd

DATA_PATH = "data/landmarks/landmark_data.csv"
SKIP = {"SPACE", "DEL", "NOTHING"}


def _normalize(X):
    pts = X.reshape(-1, 21, 3)
    pts = pts - pts[:, 0:1, :]                                    # wrist-relative
    scale = np.abs(pts).max(axis=(1, 2), keepdims=True) + 1e-8    # size-invariant
    return (pts / scale).reshape(len(X), -1)


def _build():
    df = pd.read_csv(DATA_PATH)
    labels = df["label"].astype(str).str.upper()
    X = _normalize(df.drop("label", axis=1).values)
    names = sorted(n for n in labels.unique() if n not in SKIP)
    # item profile = mean landmark vector of each sign (content-based features)
    profiles = np.array([X[(labels == n).values].mean(axis=0) for n in names])
    profiles = profiles - profiles.mean(axis=0)                   # centre so similarity is distinctive
    unit = profiles / (np.linalg.norm(profiles, axis=1, keepdims=True) + 1e-8)
    return names, unit @ unit.T                                   # cosine similarity matrix


try:
    NAMES, SIM = _build()
except FileNotFoundError:
    NAMES, SIM = [], np.zeros((0, 0))


def recommend_similar_signs(label, top_n=3):
    label = label.upper()
    if label not in NAMES:
        return []
    i = NAMES.index(label)
    order = np.argsort(-SIM[i])
    return [NAMES[j] for j in order if j != i][:top_n]


if __name__ == "__main__":
    letter = input("Letter: ")
    print("Similar signs:", recommend_similar_signs(letter))