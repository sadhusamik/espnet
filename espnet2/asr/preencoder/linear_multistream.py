#!/usr/bin/env python3

#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Linear Multistream Projection."""

from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from typeguard import check_argument_types
from typing import Tuple
import random
import numpy as np

import torch


class LinearMultistreamProjection(AbsPreEncoder):
    """Linear Multistream Projection Preencoder."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        dropout_rate,
        in_channels,
        num_channel_dropout=None
    ):
        """Initialize the module."""
        assert check_argument_types()
        super().__init__()
        self.in_channels = in_channels
        self.num_channel_dropout = num_channel_dropout

        self.lins = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.Linear(input_size, output_size),
            torch.nn.LayerNorm(output_size),
            torch.nn.Dropout(dropout_rate),
            torch.nn.ReLU(),
            torch.nn.Linear(output_size, output_size),
            torch.nn.LayerNorm(output_size),
            torch.nn.Dropout(dropout_rate),
            torch.nn.ReLU(),
            torch.nn.Linear(output_size, output_size),
            torch.nn.LayerNorm(output_size),
            torch.nn.Dropout(dropout_rate),
            torch.nn.ReLU(),
            torch.nn.Linear(output_size, output_size),
            torch.nn.LayerNorm(output_size),
            torch.nn.Dropout(dropout_rate),
            torch.nn.ReLU(),
        ) for i in range(in_channels)])

        self.out = torch.nn.Linear(in_channels * output_size, output_size)

        self.output_dim = output_size

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward."""

        outs = []
        for i in range(self.in_channels):
            out_one = self.lins[i](input[:, :, :, i])
            outs.append(out_one)

        # Outs are shaped b x t x odim
        if self.training:
            if self.num_channel_dropout is not None:
                k = np.arange(self.in_channels)
                random.shuffle(k)
                k = k[0:self.num_channel_dropout]
                for one_idx in k:
                    outs[one_idx] = outs[one_idx] * 0

        input = self.out(torch.cat(outs, dim=-1))

        return input, input_lengths  # no state in this layer

    def output_size(self) -> int:
        """Get the output size."""
        return self.output_dim
