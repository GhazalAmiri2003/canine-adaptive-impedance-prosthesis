"""Deployment template only; not trained without synchronized raw IMU + FSR windows.

Recommended outputs:
  - gait logits: walk / trot
  - continuous phase: sin(phi), cos(phi)
  - contact probability: stance / swing
  - normalized speed/loading state

The simulation uses vision-derived phase and evaluates measured timing-delay
robustness. Do not report this network as trained until raw sensor windows exist.
"""

try:
    import torch
    import torch.nn as nn
except ImportError:  # Keeps the simulation package runnable without PyTorch.
    torch = None
    nn = None


if nn is not None:
    class ResidualTCNBlock(nn.Module):
        def __init__(self, channels: int, dilation: int, dropout: float = 0.10):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
                nn.BatchNorm1d(channels), nn.ReLU(), nn.Dropout(dropout),
                nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
                nn.BatchNorm1d(channels),
            )
            self.activation = nn.ReLU()

        def forward(self, x):
            return self.activation(x + self.net(x))


    class CaninePhaseTCN(nn.Module):
        def __init__(self, in_channels: int = 7, hidden: int = 64):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv1d(in_channels, hidden, 5, padding=2), nn.BatchNorm1d(hidden), nn.ReLU())
            self.encoder = nn.Sequential(*[ResidualTCNBlock(hidden, d) for d in (1, 2, 4, 8)])
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.gait_head = nn.Linear(hidden, 2)
            self.phase_head = nn.Linear(hidden, 2)
            self.contact_head = nn.Linear(hidden, 1)
            self.speed_load_head = nn.Linear(hidden, 2)

        def forward(self, x):
            z = self.pool(self.encoder(self.stem(x))).squeeze(-1)
            return {
                "gait_logits": self.gait_head(z),
                "phase_sin_cos": self.phase_head(z),
                "contact_logit": self.contact_head(z),
                "speed_load": self.speed_load_head(z),
            }
