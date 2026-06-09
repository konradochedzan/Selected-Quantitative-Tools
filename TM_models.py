import torch
import torch.nn as nn
from typing import List, Optional
import math

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for sequences.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle both 2D and 3D inputs
        # x can be: (seq_len, d_model) or (batch, seq_len, d_model)

        if x.dim() == 2:
            # Input is 2D: (seq_len, d_model)
            seq_len, d_model = x.shape
            batch_size = 1
            x_3d = x.unsqueeze(0)  # Add batch dimension: (1, seq_len, d_model)
        elif x.dim() == 3:
            # Input is 3D: (batch, seq_len, d_model)
            batch_size, seq_len, d_model = x.shape
            x_3d = x
        else:
            raise ValueError(f"Input tensor must be 2D or 3D, got {x.dim()}D with shape {x.shape}")

        # Validate dimensions
        if d_model != self.d_model:
            raise ValueError(f"Input d_model ({d_model}) doesn't match expected d_model ({self.d_model})")

        if seq_len > self.pe.size(1):
            raise ValueError(f"Sequence length ({seq_len}) exceeds maximum length ({self.pe.size(1)})")

        # Add positional encoding
        pos_encoding = self.pe[:, :seq_len, :]
        result = x_3d + pos_encoding

        # Return in the same format as input
        if x.dim() == 2:
            return result.squeeze(0)  # Remove batch dimension
        else:
            return result


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network (TCN) for time series prediction.
    Uses dilated causal convolutions to capture long-range dependencies.
    """
    def __init__(self, input_dim: int, num_channels: Optional[List[int]] = None, kernel_size: int = 3,
                 dropout: float = 0.1, pool: str = 'last'):
        super(TemporalConvNet, self).__init__()
        if num_channels is None:
            num_channels = [64, 128, 64]
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size, 
                stride=1, dilation=dilation, padding=(kernel_size-1) * dilation,
                dropout=dropout
            ))
        
        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(num_channels[-1], 1)
        self.pool = pool

        # Scale activation to avoid ensure that each new layer begins in a setting where the outputs and gradients do
        # not vanish or blow up
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, features)
        # TCN expects (batch_size, features, seq_len)
        x = x.transpose(1, 2)
        x = self.network(x)
        if self.pool == 'avg':
            # Take average of time steps
            x = x.mean(dim=2)
        else:
            # Take the last time step
            x = x[:, :, -1]
        return self.output_layer(x)

class TemporalBlock(nn.Module):
    """Individual temporal block for TCN"""
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, stride: int, dilation: int,
                 padding:int, dropout: float = 0.1):
        super(TemporalBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.group_norm1 = nn.GroupNorm(1, n_outputs)  # GroupNorm for better stability
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.group_norm2 = nn.GroupNorm(1, n_outputs)
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.group_norm1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.group_norm2, self.dropout2)
        
        # Residual connection
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

        # Weight initialization for stable training
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class Chomp1d(nn.Module):
    """Remove padding from the end to maintain causality"""
    def __init__(self, chomp_size: int):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x


class TemporalFusionTransformer(nn.Module):
    """
    Simplified Temporal Fusion Transformer implementation.
    Focuses on key components: attention mechanisms and variable selection.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_heads: int = 8, num_layers:int = 2,
                 dropout: float = 0.1):
        super(TemporalFusionTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Variable selection network
        self.variable_selection = VariableSelectionNetwork(input_dim, hidden_dim, dropout)
        
        # Static enrichment (simplified)
        self.static_enrichment = nn.Linear(hidden_dim, hidden_dim)
        
        # Temporal processing with LSTM
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, dropout=dropout)
        
        # Multi-head attention for temporal fusion
        self.temporal_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Position-wise feed forward
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x):
        batch_size, seq_len = x.shape
        
        # Variable selection
        selected_features = self.variable_selection(x)
        
        # Static enrichment (simplified - using mean as static context)
        static_context = torch.mean(selected_features, dim=1, keepdim=True)
        static_enriched = self.static_enrichment(static_context)
        
        # Add static context to all time steps
        enriched_features = selected_features + static_enriched.expand(-1, seq_len, -1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(enriched_features)
        
        # Self-attention
        attn_out, _ = self.temporal_attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.norm1(attn_out + lstm_out)
        
        # Feed forward
        ff_out = self.feed_forward(attn_out)
        ff_out = self.norm2(ff_out + attn_out)
        
        # Output (use last time step)
        output = self.output_layer(ff_out[:, -1, :])
        
        return output

class TemporalFusionTransformer2(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_heads: int = 8, num_layers:int = 2,
                 dropout: float = 0.1):
        super(TemporalFusionTransformer2, self).__init__()
        self.hidden_dim = hidden_dim
        # Project raw inputs to hidden dimension
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        # Positional encoding for time information
        self.pos_encoding = PositionalEncoding(hidden_dim)
        # Compute mean over time and enrich static context
        self.static_enrichment = nn.Linear(hidden_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, dropout=dropout, num_layers=1)

        # Pre-norm transformer blocks: LayerNorm->Attention->residual, then LayerNorm->FFN->residual for each block
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            block = nn.ModuleDict({
                'ln1': nn.LayerNorm(hidden_dim),
                'attn': nn.MultiheadAttention(hidden_dim, num_heads,
                                              dropout=dropout, batch_first=True),
                'ln2': nn.LayerNorm(hidden_dim),
                'ff': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim)
                )
            })
            self.layers.append(block)

        self.output_layer = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
                                          nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1))

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(f"1. Initial input shape: {x.shape}")

        # Handle 2D input by adding feature dimension if needed
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (batch_size, seq_len) -> (batch_size, seq_len, 1)
            print(f"2. After unsqueeze shape: {x.shape}")

        # Project inputs into hidden space
        print(f"3. input_projection expects input size: {self.input_projection.in_features}")
        print(f"   input_projection outputs size: {self.input_projection.out_features}")
        x = self.input_projection(x)
        print(f"4. After input_projection shape: {x.shape}")

        # Positional encoding so model knows timestep
        x = self.pos_encoding(x)
        print(f"5. After pos_encoding shape: {x.shape}")

        # Now x should be 3D: (batch_size, seq_len, hidden_dim)
        if x.dim() == 3:
            batch_size, seq_len, hidden_dim = x.shape
            print(f"6. Correctly unpacked 3D: batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}")
        elif x.dim() == 2:
            batch_size, seq_len = x.shape
            print(f"6. WARNING: Still 2D after projection! batch_size={batch_size}, seq_len={seq_len}")
            print("   This indicates a problem with input_projection or pos_encoding")
            # Emergency fix: assume hidden_dim from model
            hidden_dim = self.hidden_dim
        else:
            raise ValueError(f"Unexpected tensor dimensions: {x.shape}")

        # Summary of sequence
        static = x.mean(dim=1)  # Shape depends on x dimensions
        print(f"7. Static shape after mean: {static.shape}")

        print(f"8. static_enrichment expects input size: {self.static_enrichment.in_features}")
        static_enriched = self.static_enrichment(static)
        print(f"9. Static enriched shape: {static_enriched.shape}")

        # The rest of your code...
        if x.dim() == 3:
            static_broadcasted = static_enriched.unsqueeze(1).expand(batch_size, seq_len, hidden_dim)
        else:
            # If x is still 2D, we need different broadcasting
            static_broadcasted = static_enriched.unsqueeze(1).expand(batch_size, seq_len)

        print(f"10. Static broadcasted shape: {static_broadcasted.shape}")
        x = x + static_broadcasted

        # Mix features using LSTM
        x, _ = self.lstm(x)

        # Apply each transformer block
        for block in self.layers:
            _x = block['ln1'](x)
            attention_out, _ = block['attn'](_x, _x, _x)
            x = x + attention_out

            _x = block['ln2'](x)
            ff_out = block['ff'](_x)
            x = x + ff_out

        # Use final timestep for prediction
        final_hidden = x[:, -1, :]  # Shape: (batch_size, hidden_dim)
        output = self.output_layer(final_hidden)

        return output

class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network for TFT"""
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super(VariableSelectionNetwork, self).__init__()
        
        # Flatten and project
        self.flatten_projection = nn.Linear(input_dim, hidden_dim)
        
        # Variable selection weights
        self.variable_weights = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Softmax(dim=-1)
        )
        
        # Selected variable processing
        self.selected_processing = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        if x.dim() == 2:
            batch_size, input_dim = x.shape
            seq_len = 1
            x = x.unsqueeze(1)
        else:
            batch_size, seq_len, input_dim = x.shape

        # Flatten for weight computation
        x_flat = x.view(-1, input_dim)
        
        # Compute variable selection weights
        projected = self.flatten_projection(x_flat)
        weights = self.variable_weights(projected)
        weights = weights.view(batch_size, seq_len, input_dim)
        
        # Apply weights to original features
        selected = x * weights
        
        # Process selected variables
        selected_flat = selected.view(-1, input_dim)
        processed = self.selected_processing(selected_flat)
        processed = processed.view(batch_size, seq_len, -1)
        
        return processed