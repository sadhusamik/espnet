#!/usr/bin/env python3
#  2021, Carnegie Mellon University;  Xuankai Chang
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Linear Projection."""

from espnet2.modnet_v2.projector.abs_projector import AbsProjector
from typeguard import check_argument_types
from typing import Tuple

import torch
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class ModulationProjector(AbsProjector):
    """Projecting from high dimension features to modulation domain."""

    def __init__(
            self,
            input_size: int,
            output_size: int,
            coeff_num: int,
            n_filters: int
    ):
        """Initialize the module."""
        assert check_argument_types()
        super().__init__()

        self.output_dim = output_size
        self.coeff_num = coeff_num
        self.n_filters = n_filters

        # self.convert_to_modulation_dimension = torch.nn.Linear(in_features=input_size, out_features=coeff_num)

        # This will be assuming we need 39 factor down-sampling
        # self.conv = [torch.nn.Sequential(
        #    torch.nn.Conv1d(in_channels=coeff_num, out_channels=coeff_num, padding=2, dilation=4, kernel_size=5,
        #                    stride=13),
        #    torch.nn.Tanh(),
        #    torch.nn.Conv1d(in_channels=coeff_num, out_channels=coeff_num, padding=2, dilation=4, kernel_size=5,
        #                    stride=13),
        #    torch.nn.Tanh(),
        #    torch.nn.Conv1d(in_channels=coeff_num, out_channels=coeff_num, padding=2, dilation=4, kernel_size=5,
        #                    stride=13)
        # ) for i in range(n_filters)]
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=input_size, out_channels=coeff_num, padding=2, dilation=1, kernel_size=5,
                            stride=3),
            torch.nn.Tanh(),
            torch.nn.Conv1d(in_channels=coeff_num, out_channels=coeff_num, padding=2, dilation=4, kernel_size=5,
                            stride=13))

        # self.final_linear = torch.nn.Linear(in_features=coeff_num, out_features=coeff_num)
        # self.final_linear_imag = torch.nn.Linear(in_features=input_size, out_features=coeff_num)

    def forward(
            self, input: torch.Tensor, input_lengths: torch.Tensor
    ):
        """Forward."""

        # batch_size = input.shape[0]
        # freq_dim = input.shape[-1]

        # input = torch.reshape(input, (batch_size * freq_dim, input.shape[1]))
        # input = input.unsqueeze(1)
        # input=torch.transpose(input,input.shape[-1],input.shape[1]) # batch x 256

        # input = self.convert_to_modulation_dimension(input)  # Convert from input dimension to coeff_num
        input = torch.transpose(input, 2, 1)  # batch x input_size(channels) x time
        # print(input.shape)
        # print(self.conv[0][0].device)
        # print(self.final_linear.device)
        # conv_outputs = []
        # for i in range(self.n_filters):
        conv_outputs = self.conv2(input).unsqueeze(-1)  # batch x coeff_num(channels) x time (downsampled)
        conv_outputs = [conv_outputs for i in range(self.n_filters)]

        input = torch.cat(conv_outputs, dim=-1)  # batch x coeff_num(channels) x time (downsampled) x n_filters
        input = torch.transpose(input, 1, 2)  # batch x time (downsampled) x coeff_num(channels) x n_filters
        input = torch.transpose(input, 2, 3)  # batch x time (downsampled) x n_filters x coeff_num(channels)

        # WE can treat these 'input' as modulation spectrum
        # input = torch.transpose(input, input.shape[-1], input.shape[1])  # batch x time x coeff_num(channels)

        # input = self.final_linear(input)  # batch x time x coeff_num(channels)
        # input = input[:, 0, :]
        # input = torch.reshape(input, (batch_size, -1, 20, freq_dim))
        # input_lengths = input.shape[1]
        # input = torch.cat([self.final_linear_real(input).unsqueeze(-1), self.final_linear_imag(input).unsqueeze(-1)],
        #                  dim=-1)

        return input, input_lengths

    def output_size(self) -> int:
        """Get the output size."""
        return self.output_dim
