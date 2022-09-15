#!/usr/bin/env python3
#  2021, Carnegie Mellon University;  Xuankai Chang
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Linear Projection."""

from espnet2.modnet.projector.abs_projector import AbsProjector
from typeguard import check_argument_types
from typing import Tuple

import torch


class ModulationProjector(AbsProjector):
    """Projecting from high dimension features to modulation domain."""

    def __init__(
            self,
            input_size: int,
            output_size: int,
            coeff_num: int
    ):
        """Initialize the module."""
        assert check_argument_types()
        super().__init__()

        self.output_dim = output_size

        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=20, kernel_size=5, stride=5, padding=3),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=20, out_channels=20, kernel_size=5, stride=5, padding=3),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=20, out_channels=20, kernel_size=5, stride=3, padding=3)
        )
        self.final_linear_real = torch.nn.Linear(in_features=input_size, out_features=coeff_num)
        self.final_linear_imag = torch.nn.Linear(in_features=input_size, out_features=coeff_num)

    def forward(
            self, input: torch.Tensor, input_lengths: torch.Tensor
    ):
        """Forward."""

        batch_size = input.shape[0]
        freq_dim = input.shape[-1]

        input = torch.reshape(input, (batch_size * freq_dim, input.shape[1]))
        input = input.unsqueeze(1)

        input = self.conv(input)

        # input = input[:, 0, :]
        input = torch.reshape(input, (batch_size, -1, 20, freq_dim))
        input_lengths = input.shape[1]
        input = torch.cat([self.final_linear_real(input).unsqueeze(-1), self.final_linear_imag(input).unsqueeze(-1)],
                          dim=-1)

        return input, input_lengths

    def output_size(self) -> int:
        """Get the output size."""
        return self.output_dim
