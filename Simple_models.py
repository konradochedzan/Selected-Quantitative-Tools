import torch
import torch.nn as nn
from typing import List
import math

class SimpleFeedForward(nn.Module):
    """Example feedforward model"""

    def __init__(self, input_dim, hidden_dim=200, dropout=0):
        super(SimpleFeedForward, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        return self.network(x)


class SimpleLSTM(nn.Module):
    """Example LSTM model"""

    def __init__(self, input_dim, hidden_dim=200, num_layers=2, dropout=0):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout)

        self.fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Linear(hidden_dim // 2, 1))

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])  # Use last time step output


class SimpleTransformer(nn.Module):
    def __init__(self,
                 input_dim: int,
                 model_dim: int = 128,  # divisible by nhead
                 nhead: int = 8,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 max_seq_length: int = 500):  # for positional encoding
        super().__init__()

        # project raw features to model_dim
        self.input_proj = nn.Linear(input_dim, model_dim)

        # Add positional encoding
        self.pos_encoding = PositionalEncoding(model_dim, dropout, max_seq_length)

        # vanilla encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=model_dim * 4,  # standard practice
            dropout=dropout,
            activation='gelu',  # often works better than relu for transformers
            batch_first=True  # so x is (B, T, F)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # regression head with additional processing
        self.norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(model_dim, 1)

    def forward(self, x):  # x: (B, T, input_dim)
        x = self.input_proj(x)  # (B, T, model_dim)
        x = self.pos_encoding(x)  # Add positional information
        x = self.encoder(x)  # (B, T, model_dim)
        x = self.norm(x[:, -1])  # Layer norm on last time step
        x = self.dropout(x)  # Additional dropout
        return self.fc(x)  # (B, 1)


class SimpleConvolutional(nn.Module):
    """
    1-D CNN for sequence-to-one forecasting.
    Assumes input tensor shape: (batch, seq_len, n_features)
    """

    def __init__(
            self,
            input_dim: int,  # -- n_features after any AR-lag concat
            num_channels: List[int] = [32, 64, 32],
            kernel_size: int = 5,
            dropout: float = 0.25,
            seq_length: int = 12  # 〈NEW – keep default same as pipeline〉
    ):
        super(SimpleConvolutional, self).__init__()

        layers, in_channels = [], input_dim
        for out_channels in num_channels:
            layers += [
                nn.Conv1d(in_channels, out_channels,
                          kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout)
            ]
            in_channels = out_channels

        self.conv = nn.Sequential(*layers)

        # ── force length-1 feature map so fc dim is invariant to seq_length ──
        self.pool = nn.AdaptiveAvgPool1d(1)  # ← NEW
        self.fc = nn.Linear(in_channels, 1)  # 〈in_channels == last out_channels〉

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, n_features)
        """
        x = x.transpose(1, 2)  # (batch, n_features, seq_len)
        x = self.conv(x)  # (batch, channels, L)
        x = self.pool(x)  # (batch, channels, 1)
        x = x.view(x.size(0), -1)  # (batch, channels)
        return self.fc(x).squeeze(-1)  # (batch,)


class PositionalEncoding(nn.Module):
    """Add positional encoding to input embeddings."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.transpose(0, 1)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)
