import sys
from contextlib import contextmanager
from distutils.version import LooseVersion
import logging
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from typeguard import check_argument_types

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.modnet.projector.abs_projector import AbsProjector
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class ModNet(AbsESPnetModel):
    """Self-supervised Modulation Net"""

    def __init__(
            self,
            frontend: AbsFrontend,
            encoder: AbsEncoder,
            projector: AbsProjector,
            extract_feats_in_collect_stats: bool = True,
            prediction_loss: str = 'MSE',
    ):
        assert check_argument_types()

        super().__init__()
        # note that eos is the same as sos (equivalent ID)

        self.frontend = frontend
        self.encoder = encoder
        self.projector = projector
        if prediction_loss == 'MSE':
            self.prediction_loss = torch.nn.MSELoss()
        elif prediction_loss == 'L1':
            self.prediction_loss = torch.nn.L1Loss()

        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats

    def forward(
            self,
            speech: torch.Tensor,
            speech_original: torch.Tensor,
            speech_lengths: torch.Tensor,
            speech_original_lengths: torch.Tensor,
            **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )

        """

        batch_size = speech.shape[0]

        # 1. Encoder
        encoder_out, encoder_out_lens, feats_original, dropout_mask = self.encode(speech, speech_original,
                                                                                  speech_lengths)

        loss = self._calc_predictive_loss(feats_original, encoder_out, dropout_mask)

        stats = dict(
            loss=loss.detach(),
        )

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
            self,
            speech: torch.Tensor,
            speech_original: torch.Tensor,
            speech_lengths: torch.Tensor,
            speech_original_lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if self.extract_feats_in_collect_stats:
            feats, _, feats_lengths, _ = self._extract_feats(speech, speech_original, speech_lengths)
        else:
            # Generate dummy stats if extract_feats_in_collect_stats is False
            logging.warning(
                "Generating dummy stats for feats and feats_lengths, "
                "because encoder_conf.extract_feats_in_collect_stats is "
                f"{self.extract_feats_in_collect_stats}"
            )
            feats, feats_lengths = speech, speech_lengths
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
            self, speech: torch.Tensor, speech_original: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        with autocast(False):
            # 1. Extract feats
            feats_dropout, feats_original, feats_lengths, dropout_mask = self._extract_feats(speech, speech_original,
                                                                                             speech_lengths)

        # 2. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)

        encoder_out, encoder_out_lens, _ = self.encoder(feats_dropout, feats_lengths)

        encoder_out, encoder_out_lens = self.projector(encoder_out, encoder_out_lens)

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        assert encoder_out.size(1) <= encoder_out_lens.max(), (
            encoder_out.size(),
            encoder_out_lens.max(),
        )

        return feats_dropout + encoder_out, encoder_out_lens, feats_original, dropout_mask

    def asr_feats_encode(
            self, speech: torch.Tensor, speech_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        with autocast(False):
            # 1. Extract feats
            assert speech_lengths.dim() == 1, speech_lengths.shape
            # for data-parallel
            speech = speech[:, : speech_lengths.max()]

            # There has to be a frontend that does modulation dropout
            feats_dropout, feats_lengths, = self.frontend(speech, speech_lengths)

        # 2. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)

        encoder_out, encoder_out_lens, _ = self.encoder(feats_dropout, feats_lengths)

        # encoder_out, encoder_out_lens = self.projector(encoder_out, encoder_out_lens)

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        assert encoder_out.size(1) <= encoder_out_lens.max(), (
            encoder_out.size(),
            encoder_out_lens.max(),
        )

        return encoder_out, encoder_out_lens

    def _extract_feats(
            self, speech: torch.Tensor, speech_original: torch.Tensor, speech_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        # There has to be a frontend that does modulation dropout
        feats_dropout, _, feats_lengths, dropout_mask = self.frontend(speech, speech_lengths)

        _, feats_original, _, _ = self.frontend(speech_original, speech_lengths)

        return feats_dropout, feats_original, feats_lengths, dropout_mask

    def _calc_predictive_loss(self, truth: torch.Tensor, hypo: torch.Tensor, dropout_mask: torch.Tensor):

        return self.prediction_loss(hypo * dropout_mask, truth * dropout_mask)
