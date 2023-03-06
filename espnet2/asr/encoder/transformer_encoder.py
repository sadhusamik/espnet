# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer encoder definition."""

from typing import List
from typing import Optional
from typing import Tuple

import torch
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import Conv1dLinear
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import MultiLayeredConv1d
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.subsampling import check_short_utt
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling2
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling6
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling8
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dMultichannel, Conv2dMultichannel2Channel, \
    LinearMultichannel2Channel, Conv2dSubsamplingMultichannel, Conv2dNosubsampling, LinearMultichannel, Conv2dSubsamplingMultichannel2Channel, Conv2dSubsamplingMultichannelNChannel, LinearNoSubsamplingMultichannelNChannel
from espnet.nets.pytorch_backend.transformer.subsampling import TooShortUttError
from espnet2.asr.ctc import CTC
from espnet2.asr.encoder.abs_encoder import AbsEncoder


class TransformerEncoder(AbsEncoder):
    """Transformer encoder module.

    Args:
        input_size: input dim
        output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the number of units of position-wise feed forward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        attention_dropout_rate: dropout rate in attention
        positional_dropout_rate: dropout rate after adding positional encoding
        input_layer: input layer type
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before: whether to use layer_norm before the first block
        concat_after: whether to concat attention layer's input and output
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied.
            i.e. x -> x + att(x)
        positionwise_layer_type: linear of conv1d
        positionwise_conv_kernel_size: kernel size of positionwise conv1d layer
        padding_idx: padding_idx for input_layer=embed
    """

    def __init__(
            self,
            input_size: int,
            output_size: int = 256,
            attention_heads: int = 4,
            linear_units: int = 2048,
            num_blocks: int = 6,
            dropout_rate: float = 0.1,
            in_channels: int = 80,
            num_channel_dropout: int = None,
            positional_dropout_rate: float = 0.1,
            attention_dropout_rate: float = 0.0,
            input_layer: Optional[str] = "conv2d",
            pos_enc_class=PositionalEncoding,
            normalize_before: bool = True,
            concat_after: bool = False,
            positionwise_layer_type: str = "linear",
            positionwise_conv_kernel_size: int = 1,
            padding_idx: int = -1,
            interctc_layer_idx: List[int] = [],
            interctc_use_conditioning: bool = False,
    ):
        assert check_argument_types()
        super().__init__()
        self._output_size = output_size

        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(input_size, output_size),
                torch.nn.LayerNorm(output_size),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(input_size, output_size, dropout_rate)
        elif input_layer == "conv2dnosubsampling":
            self.embed = Conv2dNosubsampling(input_size, output_size, dropout_rate)
        elif input_layer == "conv2d2":
            self.embed = Conv2dSubsampling2(input_size, output_size, dropout_rate)
        elif input_layer == "conv2d6":
            self.embed = Conv2dSubsampling6(input_size, output_size, dropout_rate)
        elif input_layer == "conv2d8":
            self.embed = Conv2dSubsampling8(input_size, output_size, dropout_rate)
        elif input_layer == "conv2dmultichannel":
            self.embed = Conv2dMultichannel(input_size, output_size, dropout_rate, in_channels=in_channels)
        elif input_layer == "conv2dsubsamplingmultichannel":
            self.embed = Conv2dSubsamplingMultichannel(input_size, output_size, dropout_rate, in_channels=in_channels)
        elif input_layer == "conv2dmultichannel2C":
            self.embed = Conv2dMultichannel2Channel(input_size, output_size, dropout_rate, in_channels=in_channels)
        elif input_layer == "conv2dsubsamplingmultichannel2C":
            self.embed = Conv2dSubsamplingMultichannel2Channel(input_size, output_size, dropout_rate, in_channels=in_channels)
        elif input_layer == "conv2dsubsamplingmultichannelnchannel":
            self.embed = Conv2dSubsamplingMultichannelNChannel(input_size, output_size, dropout_rate, in_channels=in_channels,num_channel_dropout=num_channel_dropout)
        elif input_layer == "linearnosubsamplingmultichannelnchannel":
            self.embed = LinearNoSubsamplingMultichannelNChannel(input_size, output_size, dropout_rate, in_channels=in_channels,num_channel_dropout=num_channel_dropout)
        elif input_layer == "linearmultichannel":
            self.embed = LinearMultichannel(input_size, output_size, dropout_rate, in_channels=in_channels)
        elif input_layer == "linearmultichannel2C":
            self.embed = LinearMultichannel2Channel(input_size, output_size, dropout_rate, in_channels=in_channels)
        elif input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(input_size, output_size, padding_idx=padding_idx),
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer is None:
            if input_size == output_size:
                self.embed = None
            else:
                self.embed = torch.nn.Linear(input_size, output_size)
        else:
            raise ValueError("unknown input_layer: " + input_layer)
        self.normalize_before = normalize_before
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (
                output_size,
                linear_units,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (
                output_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (
                output_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        else:
            raise NotImplementedError("Support only linear or conv1d.")
        self.encoders = repeat(
            num_blocks,
            lambda lnum: EncoderLayer(
                output_size,
                MultiHeadedAttention(
                    attention_heads, output_size, attention_dropout_rate
                ),
                positionwise_layer(*positionwise_layer_args),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)

        self.interctc_layer_idx = interctc_layer_idx
        if len(interctc_layer_idx) > 0:
            assert 0 < min(interctc_layer_idx) and max(interctc_layer_idx) < num_blocks
        self.interctc_use_conditioning = interctc_use_conditioning
        self.conditioning_layer = None

    def output_size(self) -> int:
        return self._output_size

    def forward(
            self,
            xs_pad: torch.Tensor,
            ilens: torch.Tensor,
            prev_states: torch.Tensor = None,
            ctc: CTC = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Embed positions in tensor.

        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input length (B)
            prev_states: Not to be used now.
        Returns:
            position embedded tensor and mask
        """
        if isinstance(xs_pad, list):
            masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad[0].device)
        else:
            masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)

        if self.embed is None:
            xs_pad = xs_pad
        elif (
                isinstance(self.embed, Conv2dSubsampling)
                or isinstance(self.embed, Conv2dSubsampling2)
                or isinstance(self.embed, Conv2dSubsampling6)
                or isinstance(self.embed, Conv2dSubsampling8)
                or isinstance(self.embed, Conv2dMultichannel)
                or isinstance(self.embed, Conv2dNosubsampling)
                or isinstance(self.embed, Conv2dSubsamplingMultichannel)
                or isinstance(self.embed, Conv2dMultichannel2Channel)
                or isinstance(self.embed, LinearMultichannel2Channel)
                or isinstance(self.embed, Conv2dSubsamplingMultichannel2Channel)
                or isinstance(self.embed, Conv2dSubsamplingMultichannelNChannel)
                or isinstance(self.embed, LinearNoSubsamplingMultichannelNChannel)
                or isinstance(self.embed, LinearMultichannel)
        ):
            if isinstance(xs_pad, list):
                short_status, limit_size = check_short_utt(self.embed, xs_pad[0].size(1))
            else:
                short_status, limit_size = check_short_utt(self.embed, xs_pad.size(1))
            if short_status:
                if isinstance(xs_pad, list):
                    raise TooShortUttError(
                        f"has {xs_pad[0].size(1)} frames and is too short for subsampling "
                        + f"(it needs more than {limit_size} frames), return empty results",
                        xs_pad[0].size(1),
                        limit_size,
                    )
                else:
                    raise TooShortUttError(
                        f"has {xs_pad.size(1)} frames and is too short for subsampling "
                        + f"(it needs more than {limit_size} frames), return empty results",
                        xs_pad.size(1),
                        limit_size,
                    )
            xs_pad, masks = self.embed(xs_pad, masks)
        else:
            xs_pad = self.embed(xs_pad)

        intermediate_outs = []
        if len(self.interctc_layer_idx) == 0:
            xs_pad, masks = self.encoders(xs_pad, masks)
        else:
            for layer_idx, encoder_layer in enumerate(self.encoders):
                xs_pad, masks = encoder_layer(xs_pad, masks)

                if layer_idx + 1 in self.interctc_layer_idx:
                    encoder_out = xs_pad

                    # intermediate outputs are also normalized
                    if self.normalize_before:
                        encoder_out = self.after_norm(encoder_out)

                    intermediate_outs.append((layer_idx + 1, encoder_out))

                    if self.interctc_use_conditioning:
                        ctc_out = ctc.softmax(encoder_out)
                        xs_pad = xs_pad + self.conditioning_layer(ctc_out)

        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)

        olens = masks.squeeze(1).sum(1)
        if len(intermediate_outs) > 0:
            return (xs_pad, intermediate_outs), olens, None
        return xs_pad, olens, None
