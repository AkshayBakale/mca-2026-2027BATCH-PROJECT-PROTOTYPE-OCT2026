import torch
import torch.nn as nn


class MotionNet(nn.Module):
    """Sequence classifier over landmark frames: LSTM or GRU, optionally bidirectional."""
    def __init__(self, n_classes, cell="lstm", bidirectional=False, input_dim=63, hidden=64):
        super().__init__()
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.bidirectional = bidirectional
        self.rnn = rnn(input_dim, hidden, batch_first=True, bidirectional=bidirectional)
        out_dim = hidden * (2 if bidirectional else 1)
        self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(out_dim, n_classes))

    def forward(self, x):
        _, h = self.rnn(x)
        if isinstance(h, tuple):          # LSTM returns (h_n, c_n)
            h = h[0]
        feat = torch.cat([h[-2], h[-1]], dim=1) if self.bidirectional else h[-1]
        return self.fc(feat)


class AutoEncoder(nn.Module):
    """Deep autoencoder: 63 -> 32 -> 16 -> 8 (bottleneck) -> 16 -> 32 -> 63."""
    def __init__(self, input_dim=63):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))