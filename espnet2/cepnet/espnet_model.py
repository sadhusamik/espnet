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
import soundfile
import scipy
import random

import torch
from typeguard import check_argument_types

from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder

from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class CepNet(AbsESPnetModel):
    """Speech dereverberation network"""

    def __init__(
            self,
            encoder: AbsEncoder,
            projector,
            extract_feats_in_collect_stats: bool = True,
            prediction_loss: str = 'MSE',
            srate: int = 16000,
            fduration: float = 3,
            overlap_fraction: float = 0.75,
            chunk_num: int = 5,

    ):
        assert check_argument_types()

        super().__init__()
        # note that eos is the same as sos (equivalent ID)

        self.encoder = encoder
        self.projector = projector
        self.fduration = fduration
        self.overlap_fraction = overlap_fraction
        self.srate = srate
        self.lfr = 1 / (self.overlap_fraction * self.fduration)
        self.nfft = int(fduration * srate)
        self.chunk_num = chunk_num

        if prediction_loss == 'MSE':
            self.prediction_loss = torch.nn.MSELoss()
        elif prediction_loss == 'L1':
            self.prediction_loss = torch.nn.L1Loss()

        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats

    def get_frames(self, signal: torch.Tensor) -> torch.Tensor:
        """Divide speech signal into frames.

                Args:
                    signal: (Batch, Nsamples) or (Batch, Nsample)
                Returns:
                    output: (Batch, Frame num, Frame dimension) or (Batch, Frame num, Frame dimension)
                """

        flength_samples = int(self.srate * self.fduration)
        frate_samples = int(self.srate / self.lfr)

        if flength_samples % 2 == 0:
            sp_b = int(flength_samples / 2) - 1
            sp_f = int(flength_samples / 2)
            extend = int(flength_samples / 2) - 1
        else:
            sp_b = int((flength_samples - 1) / 2)
            sp_f = int((flength_samples - 1) / 2)
            extend = int((flength_samples - 1) / 2)

        # signal = torch.nn.functional.pad(signal.unsqueeze(1), (extend, extend), mode='constant', value=0.0).squeeze(1)
        signal = torch.nn.functional.pad(signal.unsqueeze(1), (extend, extend), mode='reflect').squeeze(1)

        signal_length = signal.shape[1]

        win = torch.hamming_window(flength_samples, dtype=signal.dtype, device=signal.device)

        idx = sp_b
        frames = []
        while (idx + sp_f) < signal_length:
            frames.append(signal[:, idx - sp_b:idx + sp_f + 1].unsqueeze(1) * win)
            idx += frate_samples

        frames = torch.cat(frames, dim=1)
        return frames

    def forward_x(
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
            speech_original: (Batch, Length, ...)
            speech_lengths: (Batch, )

        """

        batch_size = speech.shape[0]
        sig_len = speech.shape[1]
        print(self.nfft)

        speech = self.get_frames(speech)  # Batch x frame_num x frame_dimension
        speech = torch.reshape(speech, (-1, speech.shape[-1]))  # Batch * frame_num x frame_dimension

        speech_original = self.get_frames(speech_original)  # Batch x frame_num x frame_dimension
        speech_original = torch.reshape(speech_original,
                                        (-1, speech_original.shape[-1]))  # Batch * frame_num x frame_dimension

        # FFT of frames
        speech = torch.fft.fft(speech, n=self.nfft)  # Batch * frame_num x nfft
        speech_original = torch.fft.fft(speech_original, n=self.nfft)  # Batch * frame_num x nfft
        speech = speech.unsqueeze(-1)  # Batch * frame_num x nfft x 1
        speech_original = speech_original.unsqueeze(-1)  # Batch * frame_num x nfft x 1

        # 1. Encoder for real and imaginary parts

        if self.chunk_size:

            # Divide computation to num_chunks along num_batch direction
            # chunk_size = int(np.ceil(speech.shape[0] / self.num_chunks))
            speech = list(torch.split(speech, split_size_or_sections=self.chunk_size, dim=0))
            encoder_out = []
            for chunk_idx in range(len(speech)):
                print(chunk_idx)
                ll = torch.Tensor([int(self.nfft / 2) + 1] * speech[chunk_idx].shape[0])
                encoder_out_chunk, _, _ = self.encoder(
                    torch.view_as_real(speech[chunk_idx][:, :int(self.nfft / 2) + 1, :]), ll)
                encoder_out_chunk = self.projector(encoder_out_chunk)
                encoder_out.append(encoder_out_chunk)

            encoder_out = torch.cat(encoder_out, dim=0)
            speech = torch.cat(speech, dim=0)
            encoder_out = torch.cat((encoder_out, torch.flip(encoder_out[:, :-2], [1])), dim=1)
            encoder_out[:, int(self.nfft / 2) + 1:, 1] = -encoder_out[:, int(self.nfft / 2) + 1:, 1]

        else:
            ll = torch.Tensor([int(self.nfft / 2) + 1] * batch_size)
            encoder_out, _, _ = self.encoder(torch.view_as_real(speech[:, :int(self.nfft / 2) + 1, :]), ll)
            encoder_out = self.projector(encoder_out)

            encoder_out = torch.cat((encoder_out, torch.flip(encoder_out[:, :-2], [1])), dim=1)
            encoder_out[:, int(self.nfft / 2) + 1:, 1] = -encoder_out[:, int(self.nfft / 2) + 1:, 1]

        encoder_out = torch.view_as_complex(encoder_out)

        # loss = self.prediction_loss(speech_original[:, :self.nfft],
        #                            torch.real(torch.fft.ifft(fft_signal / encoder_out))[:, :, 0])

        loss1 = self.prediction_loss(torch.real(speech_original), torch.real(speech / encoder_out))
        # loss1 = self.prediction_loss(torch.real(fft_signal_original), torch.real(fft_signal - 0.00000*encoder_out))
        loss2 = self.prediction_loss(torch.imag(speech_original), torch.imag(speech / encoder_out))
        # loss2 = self.prediction_loss(torch.imag(fft_signal_original), torch.imag(fft_signal - 0.00000*encoder_out))
        loss = loss2 + loss1

        stats = dict(
            loss=loss.detach(),
        )

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

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
            speech_original: (Batch, Length, ...)
            speech_lengths: (Batch, )

        """

        batch_size = speech.shape[0]
        sig_len = speech.shape[1]

        speech = self.get_frames(speech)  # Batch x frame_num x frame_dimension
        speech = torch.reshape(speech, (-1, speech.shape[-1]))  # Batch * frame_num x frame_dimension

        speech_original = self.get_frames(speech_original)  # Batch x frame_num x frame_dimension
        speech_original = torch.reshape(speech_original,
                                        (-1, speech_original.shape[-1]))  # Batch * frame_num x frame_dimension

        # choose num_chunk frames randomly
        a = np.arange(speech.shape[0])
        random.shuffle(a)
        a = a[:self.chunk_num]
        speech = speech[a, :]
        speech_original = speech_original[a, :]

        # if sig_len > self.nfft:
        #    rand_loc = int(np.random.choice(sig_len - self.nfft - 1, 1))
        #    speech = speech[:, rand_loc:rand_loc + self.nfft]
        #    speech_original = speech_original[:, rand_loc:rand_loc + self.nfft]

        speech = torch.fft.fft(speech, n=self.nfft)  # Batch * frame_num x nfft
        speech_original = torch.fft.fft(speech_original, n=self.nfft)  # Batch * frame_num x nfft

        # 1. Encoder for real and imaginary parts
        ll = torch.Tensor([int(self.nfft / 2) + 1] * speech.shape[0])
        encoder_out, _, _ = self.encoder(torch.view_as_real(speech[:, :int(self.nfft / 2) + 1]), ll)

        encoder_out = self.projector(encoder_out)
        encoder_out = torch.cat((encoder_out, torch.flip(encoder_out[:, :-2], [1])), dim=1)
        encoder_out[:, int(self.nfft / 2) + 1:, 1] = -encoder_out[:, int(self.nfft / 2) + 1:, 1]

        encoder_out = torch.view_as_complex(encoder_out)

        # loss = self.prediction_loss(speech_original[:, :self.nfft],
        #                            torch.real(torch.fft.ifft(fft_signal / encoder_out))[:, :, 0])

        loss1 = self.prediction_loss(torch.real(speech_original), torch.real(speech / encoder_out))
        loss2 = self.prediction_loss(torch.imag(speech_original), torch.imag(speech / encoder_out))

        loss = loss2 + loss1

        stats = dict(
            loss=loss.detach(),
        )

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def clean_up(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
    ):
        """Frontend + Encoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )

        """

        batch_size = speech.shape[0]
        sig_len = speech.shape[1]

        speech = self.get_frames(speech)  # Batch x frame_num x frame_dimension
        frame_num = speech.shape[1]
        speech = torch.reshape(speech, (-1, speech.shape[-1]))  # Batch * frame_num x frame_dimension

        speech = torch.fft.fft(speech, n=self.nfft)  # Batch * frame_num x nfft

        # 1. Encoder for real and imaginary parts
        ll = torch.Tensor([int(self.nfft / 2) + 1] * speech.shape[0])
        encoder_out, _, _ = self.encoder(torch.view_as_real(speech[:, :int(self.nfft / 2) + 1]), ll)

        encoder_out = self.projector(encoder_out)
        encoder_out = torch.cat((encoder_out, torch.flip(encoder_out[:, :-2], [1])), dim=1)
        encoder_out[:, int(self.nfft / 2) + 1:, 1] = -encoder_out[:, int(self.nfft / 2) + 1:, 1]

        encoder_out = torch.view_as_complex(encoder_out)

        speech = torch.real(torch.fft.ifft(speech / encoder_out))  # Batch * frame_num x nfft
        speech = torch.reshape(speech, (batch_size, frame_num, -1))  # Batch x frame_num x frame_dimension

        return encoder_out, speech

    def collect_feats(
            self,
            speech: torch.Tensor,
            speech_original: torch.Tensor,
            speech_lengths: torch.Tensor,
            speech_original_lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        # Generate dummy stats if extract_feats_in_collect_stats is False
        logging.warning(
            "Generating dummy stats for feats and feats_lengths, "
            "because encoder_conf.extract_feats_in_collect_stats is "
            f"{self.extract_feats_in_collect_stats}"
        )
        feats, feats_lengths = speech, speech_lengths
        return {"feats": feats, "feats_lengths": feats_lengths}
