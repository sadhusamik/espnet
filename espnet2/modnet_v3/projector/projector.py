#!/usr/bin/env python3
#  2021, Carnegie Mellon University;  Xuankai Chang
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Linear Projection."""

from espnet2.modnet.projector.abs_projector import AbsProjector
from typeguard import check_argument_types
from typing import Tuple

import torch


class LinearProjector(AbsProjector):
    """Linear Projection Preencoder."""

    def __init__(
            self,
            input_size: int = 256,
            output_size: int = 20,
    ):
        """Initialize the module."""
        assert check_argument_types()
        super().__init__()

        self.output_dim = output_size
        self.linear_out = torch.nn.Linear(input_size, output_size)

    def forward(
            self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward."""
        output = self.linear_out(input)
        return output, input_lengths  # no state in this layer

    def output_size(self) -> int:
        """Get the output size."""
        return self.output_dim
