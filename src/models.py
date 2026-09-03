"""PyTorch model definitions used by the final ensemble."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class Chomp1d(nn.Module):
    """Remove right padding so each convolution remains causal."""

    def __init__(self, amount: int) -> None:
        super().__init__()
        self.amount = amount

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.amount] if self.amount else x


class TemporalBlock(nn.Module):
    """Two residual causal convolutions at one dilation level."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(
                input_channels,
                output_channels,
                kernel,
                padding=padding,
                dilation=dilation,
            ),
            Chomp1d(padding),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                output_channels,
                output_channels,
                kernel,
                padding=padding,
                dilation=dilation,
            ),
            Chomp1d(padding),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.skip = (
            nn.Conv1d(input_channels, output_channels, 1)
            if input_channels != output_channels
            else nn.Identity()
        )
        self.norm = nn.LayerNorm(output_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(
            (self.net(x) + self.skip(x)).transpose(1, 2)
        ).transpose(1, 2)


class Direct336TCN(nn.Module):
    """Direct 336-hour covariate-aware TCN used for all three seeds."""

    def __init__(
        self,
        hist_size: int,
        future_size: int,
        n_series: int,
        target_idx: int,
        horizon: int = 336,
        channels: int = 96,
        levels: int = 7,
        kernel_size: int = 3,
        dropout: float = 0.15,
        series_dim: int = 16,
    ) -> None:
        super().__init__()
        self.target_idx = target_idx
        self.horizon = horizon

        history_blocks = []
        for level in range(levels):
            history_blocks.append(
                TemporalBlock(
                    hist_size if level == 0 else channels,
                    channels,
                    kernel_size,
                    2**level,
                    dropout,
                )
            )
        self.history_tcn = nn.Sequential(*history_blocks)
        self.future_input = nn.Sequential(
            nn.Linear(future_size, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
        )
        self.future_tcn = nn.Sequential(
            TemporalBlock(channels, channels, 3, 1, dropout),
            TemporalBlock(channels, channels, 3, 2, dropout),
            TemporalBlock(channels, channels, 3, 4, dropout),
        )
        self.series = nn.Embedding(n_series, series_dim)
        self.lead = nn.Embedding(horizon, channels)
        self.anchor_projection = nn.Sequential(
            nn.Linear(2, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
        )
        joined_size = channels * 4 + series_dim
        self.fusion = nn.Sequential(
            nn.Linear(joined_size, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(channels),
        )
        self.anchor_gate = nn.Linear(channels, 1)
        self.correction = nn.Sequential(
            nn.Linear(channels, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        x_hist: torch.Tensor,
        x_future: torch.Tensor,
        series_idx: torch.Tensor,
    ) -> torch.Tensor:
        encoded = self.history_tcn(x_hist.transpose(1, 2))
        context = 0.7 * encoded[:, :, -1] + 0.3 * encoded.mean(dim=2)
        context = context[:, None, :].expand(-1, x_future.shape[1], -1)

        future = self.future_input(x_future).transpose(1, 2)
        future = self.future_tcn(future).transpose(1, 2)
        lead = self.lead(
            torch.arange(x_future.shape[1], device=x_future.device)
        )[None].expand(x_hist.shape[0], -1, -1)
        series = self.series(series_idx)[:, None, :].expand(
            -1, x_future.shape[1], -1
        )

        weekly = x_hist[:, -168:, self.target_idx].repeat(
            1, int(np.ceil(x_future.shape[1] / 168))
        )[:, : x_future.shape[1]]
        last = x_hist[:, -1:, self.target_idx].expand(-1, x_future.shape[1])
        anchors = self.anchor_projection(torch.stack([weekly, last], dim=-1))
        fused = self.fusion(
            torch.cat([context, future, lead, anchors, series], dim=-1)
        )
        gate = torch.sigmoid(self.anchor_gate(fused)).squeeze(-1)
        base = gate * weekly + (1.0 - gate) * last
        return base + self.correction(fused).squeeze(-1)


def get_tft_class():
    """Create the checkpoint-compatible TFT subclass lazily."""
    from pytorch_forecasting import TemporalFusionTransformer

    class TFTFullOneCycle(TemporalFusionTransformer):
        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.hparams.learning_rate,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.1,
                div_factor=10.0,
                final_div_factor=100.0,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }

    return TFTFullOneCycle

