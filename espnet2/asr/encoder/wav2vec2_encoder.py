# Copyright 2021 Xuankai Chang
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder definition."""
import contextlib
import copy
import sys

import numpy as np
from filelock import FileLock
import logging
import os
from typing import Optional, Tuple

import torch
from filelock import FileLock
from typeguard import check_argument_types

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm


class FairSeqWav2Vec2Encoder(AbsEncoder):
    """FairSeq Wav2Vec2 encoder module.

    Args:
        input_size: input dim
        output_size: dimension of attention
        w2v_url: url to Wav2Vec2.0 pretrained model
        w2v_dir_path: directory to download the Wav2Vec2.0 pretrained model.
        normalize_before: whether to use layer_norm before the first block
        finetune_last_n_layers: last n layers to be finetuned in Wav2Vec2.0
                                0 means to finetune every layer if freeze_w2v=False.
    """

    def __init__(
            self,
            input_size: int,
            w2v_url: str,
            w2v_dir_path: str = "./",
            output_size: int = 768,
            normalize_before: bool = False,
            start_finetune_after: int = 0,
            freeze: bool = False,
            freeze_encoder: bool = True,
            freeze_first_n: int = None,
    ):
        assert check_argument_types()
        super().__init__()

        if w2v_url != "":
            try:
                import fairseq
                from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Model
            except Exception as e:
                print("Error: FairSeq is not properly installed.")
                print(
                    "Please install FairSeq: cd ${MAIN_ROOT}/tools && make fairseq.done"
                )
                raise e

        self.w2v_model_path = download_w2v(w2v_url, w2v_dir_path)

        self._output_size = output_size

        models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [self.w2v_model_path],
            arg_overrides={"data": w2v_dir_path},
        )
        model = models[0]

        if not isinstance(model, Wav2Vec2Model):
            try:
                model = model.w2v_encoder.w2v_model
            except Exception as e:
                print(
                    "Error: pretrained models should be within: "
                    "'Wav2Vec2Model, Wav2VecCTC' classes, etc."
                )
                raise e

        self.encoders = model

        self.pretrained_params = copy.deepcopy(model.state_dict())

        self.normalize_before = normalize_before
        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)

        if model.cfg.encoder_embed_dim != output_size:
            # TODO(xkc09): try LSTM
            self.output_layer = torch.nn.Sequential(
                torch.nn.Linear(model.cfg.encoder_embed_dim, output_size),
            )
        else:
            self.output_layer = None

        self.start_finetune_after = start_finetune_after
        logging.info("Will start fine-tuning wav2vec after {:d} steps".format(self.start_finetune_after))
        self.register_buffer("num_updates", torch.LongTensor([0]))
        self.freeze = freeze
        if self.freeze:
            logging.info("Wav2vec parameters are frozen.")
        else:
            logging.info("Updating wav2vec parameters")
        if freeze_encoder:
            logging.info("Freezing CNN encoder - will only update transformer layers")
            self.encoders.feature_extractor.requires_grad_(False)
        if freeze_first_n:
            self.encoders.post_extract_proj.requires_grad_(False)
            self.encoders.quantizer.requires_grad_(False)
            self.encoders.project_q.requires_grad_(False)
            for i in range(freeze_first_n):
                self.encoders.encoder.layers[i].requires_grad_(False)

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward FairSeqWav2Vec2 Encoder.

        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input length (B)
            prev_states: Not to be used now.
        Returns:
            position embedded tensor and mask
        """
        masks = make_pad_mask(ilens).to(xs_pad.device)

        ft = self.start_finetune_after <= self.num_updates  # Sets True when num_updates is above threshold
        if self.num_updates <= self.start_finetune_after:
            self.num_updates += 1
        elif ft and self.num_updates == self.start_finetune_after + 1:
            self.num_updates += 1
            logging.info("Start fine-tuning wav2vec parameters!")

        if self.freeze or not ft :
            with torch.no_grad():  # if not ft else contextlib.nullcontext():
                enc_outputs = self.encoders(
                    xs_pad,
                    masks,
                    mask=False,
                    features_only=True,
                )
        else:
            with contextlib.nullcontext():
                enc_outputs = self.encoders(
                    xs_pad,
                    masks,
                    mask=False,
                    features_only=True,
                )

        xs_pad = enc_outputs["x"]  # (B,T,C),
        bs = xs_pad.shape[0]
        if enc_outputs["padding_mask"] is not None:
            masks = enc_outputs["padding_mask"]  # (B, T)
            olens = (~masks).sum(dim=1)  # (B)
        else:
            olens = torch.IntTensor([xs_pad.shape[1]]).repeat(bs).to(xs_pad.device)

        if self.output_layer is not None:
            xs_pad = self.output_layer(xs_pad)

        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)

        return xs_pad, olens, None

    def reload_pretrained_parameters(self):
        self.encoders.load_state_dict(self.pretrained_params)
        logging.info("Pretrained Wav2Vec model parameters reloaded!")


def download_w2v(model_url, dir_path):
    os.makedirs(dir_path, exist_ok=True)

    model_name = model_url.split("/")[-1]
    model_path = os.path.join(dir_path, model_name)

    dict_url = "https://dl.fbaipublicfiles.com/fairseq/wav2vec/dict.ltr.txt"
    dict_path = os.path.join(dir_path, dict_url.split("/")[-1])

    with FileLock(model_path + ".lock"):
        if not os.path.exists(model_path):
            torch.hub.download_url_to_file(model_url, model_path)
            torch.hub.download_url_to_file(dict_url, dict_path)
            logging.info(f"Wav2Vec model downloaded {model_path}")
        else:
            logging.info(f"Wav2Vec model {model_path} already exists.")

    return model_path
