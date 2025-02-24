import sys
from contextlib import contextmanager
from distutils.version import LooseVersion
import logging
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
import numpy as np

import torch
from typeguard import check_argument_types

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.modnet_v2.projector.abs_projector import AbsProjector
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class ModNet_v2(AbsESPnetModel):
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
        # speech = speech[:, :, 0]

        batch_size = speech.shape[0]

        # 1. Encoder
        encoder_out, encoder_out_lens, feats_original, feats_dropout = self.encode(speech, speech_original,
                                                                                   speech_lengths)
        loss = self._calc_predictive_loss(feats_original['feats_original'], encoder_out,
                                          feats_dropout['random_frame_idx'])

        stats = dict(
            loss=loss.detach(),
        )

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def _calc_predictive_loss(self, feats_original, encoder_out, random_frame_idx):

        # han_weight = torch.hann_window(int(self.frontend.fduration * self.frontend.frate), dtype=encoder_out.dtype,
        #                               device=encoder_out.device)
        # ham_weight = torch.hamming_window(int(self.frontend.fduration * self.frontend.frate), dtype=encoder_out.dtype,
        #                                  device=encoder_out.device)

        if self.frontend.complex_modulation:
            encoder_out = torch.fft.fft(torch.view_as_complex(encoder_out),
                                        1 * int(self.frontend.fduration * self.frontend.frate))
        else:
            encoder_out = torch.fft.fft(encoder_out, 2 * int(self.frontend.fduration * self.frontend.frate))

        encoder_out = encoder_out[:, :, :, 0:int(
            self.frontend.fduration * self.frontend.frate)]  # (batch x num_frames x n_filters x int(self.fduration * self.frate))
        # encoder_out = torch.abs(torch.exp(encoder_out))
        # encoder_out = encoder_out * han_weight / ham_weight
        # encoder_out = torch.transpose(encoder_out, 2, 3)

        # encoder_out=encoder_out[]

        # print(encoder_out.shape)
        # print(feats_original.shape)
        # sys.stdout.flush()

        num_batch = encoder_out.shape[0]
        batch_idx = np.arange(num_batch)
        loss = torch.Tensor([0])
        loss = loss.to(encoder_out.device)
        count = 0
        print(encoder_out.shape)
        print(feats_original.shape)
        for p, q in zip(batch_idx, random_frame_idx):
            #for freq_band in range(feats_original.shape[2]):
            count += 1
            loss += self.prediction_loss(feats_original[p, q, :, :], encoder_out[p, q, :, :])

        return loss / count

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
            self, speech: torch.Tensor, speech_original: torch.Tensor, speech_lengths: torch.Tensor):
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        with autocast(False):
            # 1. Extract feats
            feats_dropout, feats_original, feats_lengths = self._extract_feats(speech, speech_original, speech_lengths)

        # 2. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)

        encoder_out, encoder_out_lens, _ = self.encoder(feats_dropout['feats'], feats_lengths)

        encoder_out, encoder_out_lens = self.projector(encoder_out, encoder_out_lens)

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        # assert encoder_out.size(1) <= encoder_out_lens.max(), (
        #    encoder_out.size(),
        #    encoder_out_lens.max(),
        # )

        return encoder_out, encoder_out_lens, feats_original, feats_dropout

    def analyze(
            self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ):
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        with autocast(False):
            # 1. Extract feats
            feats_dropout, feats_original, feats_lengths = self._extract_feats(speech, speech,
                                                                               speech_lengths)

        # 2. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)

        encoder_out, encoder_out_lens, _ = self.encoder(feats_dropout['feats'], feats_lengths)

        encoder_out, encoder_out_lens = self.projector(encoder_out, encoder_out_lens)

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        assert encoder_out.size(1) <= encoder_out_lens.max(), (
            encoder_out.size(),
            encoder_out_lens.max(),
        )

        return {'feats_dropout': feats_dropout, 'feats_original': feats_original, 'encoder_out': encoder_out,
                'dropout_mask': dropout_mask}

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
    ):
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        # There has to be a frontend that does modulation dropout
        feats_dropout, _ = self.frontend(speech, speech_lengths)

        feats_original, olens = self.frontend(speech_original, speech_lengths)

        return feats_dropout, feats_original, olens
