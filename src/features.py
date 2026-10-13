import numpy as np


def normalize(X):
    """Raw MediaPipe landmarks (n, 63) -> wrist-relative, size-normalised (n, 63)."""
    X = np.asarray(X, dtype=np.float32).reshape(-1, 21, 3)
    X = X - X[:, 0:1, :]                                      # position-independent
    scale = np.abs(X).max(axis=(1, 2), keepdims=True) + 1e-8  # distance-independent
    return (X / scale).reshape(-1, 63)