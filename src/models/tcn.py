"""
src/models/tcn.py
-----------------
Temporal Convolutional Network (TCN) — Stage 1 of the Anomaly Detection Agent.

Paper Section 3.7, Equation (9):
  "A Temporal Convolutional Network (TCN) with 4 layers, 64 channels, and
   kernel size 3 encodes a lookback window of 30 time steps into a latent
   representation h_t ∈ R^{|E| × 128}."

Architecture follows Bai et al. (2018) "An Empirical Evaluation of Generic
Convolutional and Recurrent Networks for Sequence Modeling."

Key design choices:
  - Dilated causal convolutions prevent future information leakage.
  - Dilation schedule [1, 2, 4, 8] grows the receptive field to 30+ steps
    with only 4 layers and kernel size 3.
  - Residual connections stabilise deep TCN training.
  - Per-edge processing: input has shape [batch, |E|, T, d_feat], output
    has shape [batch, |E|, latent_dim].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class CausalConv1d(nn.Module):
    """
    1-D causal (left-padded) dilated convolution.
    Ensures output at position t only depends on inputs at positions ≤ t.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int = 1):
        super().__init__()
        # Left padding only so future time steps are never visible
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=0,           # Manual left-padding below
            dilation=dilation,
        )
        self._causal_padding = self.padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, channels, time]
        x = F.pad(x, (self._causal_padding, 0))  # Left-pad
        return self.conv(x)


class TemporalBlock(nn.Module):
    """
    One TCN residual block: two causal dilated convolutions + skip connection.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # 1×1 residual projection if channel dimensions differ
        self.residual_proj = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, in_channels, time]
        residual = self.residual_proj(x)

        out = self.conv1(x)
        # LayerNorm expects [..., channels] — transpose, norm, transpose back
        out = self.norm1(out.transpose(1, 2)).transpose(1, 2)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.norm2(out.transpose(1, 2)).transpose(1, 2)
        out = self.relu(out)
        out = self.dropout(out)

        return self.relu(out + residual)


class TemporalConvNet(nn.Module):
    """
    Full TCN encoder.

    Input:  x ∈ R^{batch × |E| × T × d_feat}
    Output: h ∈ R^{batch × |E| × latent_dim}

    The |E| edge dimension is processed independently (weight-shared),
    which allows the same model to handle varying network sizes.
    """

    def __init__(self,
                 input_dim: int = 12,        # d_feat
                 num_channels: int = 64,     # Paper: 64 channels per layer
                 num_layers: int = 4,        # Paper: 4 layers
                 kernel_size: int = 3,       # Paper: kernel size 3
                 latent_dim: int = 128,      # h_t dimensionality
                 dilations: List[int] = None,
                 dropout: float = 0.1):
        super().__init__()

        if dilations is None:
            # Standard TCN dilation schedule (Assumption A5): 1, 2, 4, 8
            dilations = [2 ** i for i in range(num_layers)]
        assert len(dilations) == num_layers, \
            "len(dilations) must equal num_layers"

        # Build stack of TemporalBlocks
        layers = []
        in_ch = input_dim
        for d in dilations:
            layers.append(
                TemporalBlock(in_ch, num_channels, kernel_size, d, dropout)
            )
            in_ch = num_channels
        self.network = nn.Sequential(*layers)

        # Project final channel dimension to latent_dim
        self.output_proj = nn.Linear(num_channels, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Sensor tensor [batch, num_edges, T, d_feat]
               T = lookback_window (30 steps), d_feat = 12

        Returns:
            h: Latent edge features [batch, num_edges, latent_dim]
        """
        batch, num_edges, T, d_feat = x.shape

        # Flatten edge and batch dims to process all edges in parallel
        # Shape: [batch * num_edges, d_feat, T]
        x_flat = x.reshape(batch * num_edges, T, d_feat)
        x_flat = x_flat.permute(0, 2, 1)   # → [B*E, d_feat, T]

        # TCN forward pass
        h_flat = self.network(x_flat)       # → [B*E, num_channels, T]

        # Take the final time step's features
        h_flat = h_flat[:, :, -1]          # → [B*E, num_channels]

        # Project to latent_dim
        h_flat = self.output_proj(h_flat)  # → [B*E, latent_dim]

        # Restore edge dimension
        h = h_flat.reshape(batch, num_edges, -1)  # → [B, E, latent_dim]
        return h
