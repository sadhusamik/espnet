# Copyright 2021 Xuankai Chang
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder definition."""
import contextlib
import copy
import numpy as np
from filelock import FileLock
import logging
import os
from typing import Optional
from typing import Tuple
import yaml

import torch
from typeguard import check_argument_types


from espnet2.train.class_choices import ClassChoices

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.encoder.conformer_encoder import ConformerEncoder
from espnet2.asr.encoder.hubert_encoder import FairseqHubertEncoder
from espnet2.asr.encoder.hubert_encoder import FairseqHubertPretrainEncoder
from espnet2.asr.encoder.rnn_encoder import RNNEncoder
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.asr.encoder.contextual_block_transformer_encoder import (
    ContextualBlockTransformerEncoder,  # noqa: H301
)

from espnet2.asr.encoder.vgg_rnn_encoder import VGGRNNEncoder
from espnet2.asr.encoder.wav2vec2_encoder import FairSeqWav2Vec2Encoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.frontend.robust import RobustFrontend
from espnet2.modnet.projector.abs_projector import AbsProjector
from espnet2.modnet.projector.projector import LinearProjector

from espnet2.modnet.espnet_model import ModNet


frontend_choices = ClassChoices(
    name="frontend",
    classes=dict(
        robust=RobustFrontend,
    ),
    type_check=AbsFrontend,
    default="robust",
)

encoder_choices = ClassChoices(
    "encoder",
    classes=dict(
        conformer=ConformerEncoder,
        transformer=TransformerEncoder,
        contextual_block_transformer=ContextualBlockTransformerEncoder,
        vgg_rnn=VGGRNNEncoder,
        rnn=RNNEncoder,
        wav2vec2=FairSeqWav2Vec2Encoder,
        hubert=FairseqHubertEncoder,
        hubert_pretrain=FairseqHubertPretrainEncoder,
    ),
    type_check=AbsEncoder,
    default="rnn",
)
projector_choices = ClassChoices(
    "projector",
    classes=dict(
        linear=LinearProjector,
    ),
    type_check=AbsProjector,
    default="linear",
)


class ModnetEncoder(AbsEncoder):
    """Modulation Network encoder module.

    Args:
        input_size: Input Size
        modnet_conf: Modulation Network configuration
        modnet_model: Modulation Network Model
        freeze_model: Freeze the encoder
        modulation_dropout: Overwrite Modulation dropout True/False
        dropout_range_hz: Overwrite modulation dropout Range
        dropout_width_hz: Overwirte modulation dropout width
        dropout_num: Overwrite modulation dropout numbers
        dropout_frame_num: Overwrite number for frames to do modulation dropout on

    """

    def __init__(
            self,
            input_size: int,
            modnet_conf: str,
            modnet_model: str = None,
            freeze_model: bool = False,
            modulation_dropout: bool = False,
            dropout_range_hz: str = '1,20',
            dropout_width_hz: float = 1,
            dropout_num: int = 3,
            dropout_frame_num: int = 1,
    ):
        assert check_argument_types()
        super().__init__()

        args = self.yaml2dict(modnet_conf)
        # Change masking configs
        args['frontend_conf']['return_nondropout_spectrogram'] = False
        args['frontend_conf']['return_dropout_mask'] = False
        args['frontend_conf']['modulation_dropout'] = modulation_dropout
        args['frontend_conf']['dropout_range_hz'] = dropout_range_hz
        args['frontend_conf']['dropout_width_hz'] = dropout_width_hz
        args['frontend_conf']['dropout_num'] = dropout_num
        args['frontend_conf']['dropout_frame_num'] = dropout_frame_num
        args['projector'] = 'linear'

        self.feat_size = None

        self.modnet = self.build_modnet_model(args)
        if modnet_model:
            logging.info('Loading modnet model from checkpoint {:s}'.format(modnet_model))
            load_ckpt = torch.load(modnet_model, map_location=torch.device('cpu'))
            self.pretrained_params = copy.deepcopy(load_ckpt)
        else:
            self.pretrained_params = None
        if freeze_model:
            logging.info('Freezing modnet model parameters')
            self.modnet.requires_grad_(False)

        self.frontend = None

    def output_size(self) -> int:
        return self.feat_size

    def yaml2dict(self, yaml_file):
        with open(yaml_file) as f:
            dataMap = yaml.safe_load(f)
        return dataMap

    class_choices_list = [
        # --frontend and --frontend_conf
        frontend_choices,
        # --specaug and --specaug_conf
        encoder_choices,

        projector_choices,
    ]
    def build_modnet_model(self, args):

        # 1. frontend

        # Extract features in the model
        frontend_class = frontend_choices.get_class(args['frontend'])
        frontend = frontend_class(**args['frontend_conf'])
        input_size = frontend.output_size()

        # 4. Encoder
        encoder_class = encoder_choices.get_class(args['encoder'])
        encoder = encoder_class(input_size=input_size, **args['encoder_conf'])

        # Projection
        projector_class = projector_choices.get_class(args['projector'])
        encoder_output_size = encoder.output_size()
        projector = projector_class(input_size=encoder_output_size, output_size=input_size)

        # 8. Build model
        model = ModNet(
            frontend=frontend,
            encoder=encoder,
            projector=projector,
            **args['model_conf'],
        )
        self.feat_size = encoder_output_size
        return model

    def forward(
            self,
            xs_pad: torch.Tensor,
            ilens: torch.Tensor,
            prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward Modnet Encoder.

        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input length (B)
            prev_states: Not to be used now.
        Returns:
            position embedded tensor and mask
        """

        # 1. Encode features
        xs_pad, olens = self.modnet.asr_feats_encode(xs_pad, ilens)

        # 3. [Multi channel case]: Select a channel
        if xs_pad.dim() == 4:
            # h: (B, T, C, F) -> h: (B, T, F)
            if self.training:
                # Select 1ch randomly
                ch = np.random.randint(xs_pad.size(2))
                input_spec = xs_pad[:, :, ch, :]
            else:
                # Use the first channel
                input_spec = xs_pad[:, :, 0, :]

        return xs_pad, olens, None

    def reload_pretrained_parameters(self):
        if self.pretrained_params:
            self.modnet.load_state_dict(self.pretrained_params)
            logging.info("Pretrained modnet model parameters reloaded!")
        else:
            logging.info("No pretrained model provided...")



