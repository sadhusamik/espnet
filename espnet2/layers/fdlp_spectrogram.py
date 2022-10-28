import random
from distutils.version import LooseVersion
from typing import Optional
from typing import Tuple
import logging

import torch
from typeguard import check_argument_types
import numpy as np
from random import randrange
import pickle as pkl
import sys
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from torch.profiler import profile, record_function, ProfilerActivity
import time
import contextlib

is_torch_1_9_plus = LooseVersion(torch.__version__) >= LooseVersion("1.9.0")

is_torch_1_7_plus = LooseVersion(torch.__version__) >= LooseVersion("1.7")


class fdlp_spectrogram(torch.nn.Module):
    def __init__(
            self,
            n_filters: int = 20,
            coeff_num: int = 80,
            coeff_range: str = '1,80',
            order: int = 80,
            fduration: float = 1.5,
            frate: int = 125,
            overlap_fraction: float = 0.5,
            srate: int = 16000,
            do_bwe: bool = False,
            bwe_factor: float = 0.8,
            bwe_iter_num: int = 1,
            precision_lpc: bool = True,
            update_fbank: bool = False,
            update_lifter: bool = False,
            update_lifter_multiband: bool = False,
            use_complex_lifter: bool = False,
            scale_lifter_gradient: float = None,
            boost_lifter_lr: float = 1,
            freeze_lifter_finetune_updates: int = None,
            initialize_lifter: str = None,
            lifter_nonlinear_transformation: str = None,
            complex_modulation: bool = False,
            num_chunks: int = None,
            concat_utts_before_frames: bool = False,
            feature_batch: int = None,
            fbank_config: str = '1,1,2.5',  # om_w,alpha,beta
            spectral_substraction_vector: str = None,
            dereverb_whole_sentence: bool = False,
            online_normalize: bool = False,
            device: str = 'auto',
    ):
        assert check_argument_types()
        super().__init__()
        # if device == 'auto':
        #    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # else:
        self.dereverb_whole_sentence = dereverb_whole_sentence
        self.device = torch.device("cpu")
        self.concat_utts_before_frames = concat_utts_before_frames
        self.precision_lpc = precision_lpc
        self.datatype = torch.float32
        self.do_bwe = do_bwe
        self.bwe_factor = bwe_factor
        self.bwe_iter_num = bwe_iter_num
        self.n_filters = n_filters
        self.coeff_num = coeff_num
        coeff_range = coeff_range.split(',')
        self.lowpass = int(coeff_range[0])
        self.highpass = int(coeff_range[1])
        self.order = order
        self.fduration = fduration
        self.frate = frate
        self.overlap_fraction = 1 - overlap_fraction
        self.srate = srate
        self.lfr = 1 / (self.overlap_fraction * self.fduration)
        self.fbank_config = [float(x) for x in fbank_config.split(',')]
        mask = []
        for i in range(coeff_num):
            if i >= self.lowpass and i <= self.highpass:
                mask.append(1)
            else:
                mask.append(0)
        mask = torch.tensor(np.asarray(mask), dtype=self.datatype, device=self.device)
        # mask = torch.tensor(np.asarray(mask), dtype=self.datatype)
        self.mask = mask
        self.cut = int(np.round(
            self.fduration * self.frate))  # int(self.fduration * self.frate) #int(np.round(self.fduration * self.frate))
        self.cut_half = int(np.round(self.fduration * self.frate / 2))  # int(np.round(self.fduration * self.frate / 2))
        self.cut_overlap = int(np.round(self.fduration * self.frate * self.overlap_fraction))
        self.update_lifter_multiband = update_lifter_multiband
        self.update_lifter = update_lifter
        self.use_complex_lifter = use_complex_lifter
        if self.use_complex_lifter:
            logging.info('Using complex valued lifters')
        self.complex_modulation = complex_modulation
        self.scale_lifter_gradient = scale_lifter_gradient
        self.freeze_lifter_finetune_updates = freeze_lifter_finetune_updates

        # self.num_updates = 0
        # self.boost_lifter_lr = boost_lifter_lr
        self.register_buffer("boost_lifter_lr", torch.Tensor([boost_lifter_lr]))
        self.register_buffer("num_updates", torch.LongTensor([0]))
        logging.info('Boosting lifter learning rate by {}'.format(self.boost_lifter_lr.data))

        if num_chunks:
            self.num_chunks = num_chunks
        else:
            self.num_chunks = None
        if lifter_nonlinear_transformation:
            if lifter_nonlinear_transformation == 'relu':
                logging.info('Using ReLU transformed lifters')
                self.lnlt = torch.nn.ReLU()
            elif lifter_nonlinear_transformation == 'softmax':
                logging.info('Using softmax transformed lifters')
                self.lnlt = torch.nn.Softmax(dim=-1)
            else:
                logging.info('Not using any transformation for lifters')
                self.lnlt = None
        else:
            logging.info('Not using any transformation for lifters')
            self.lnlt = None

        if update_fbank:
            self.fbank = torch.nn.Parameter(
                self.initialize_filterbank(self.n_filters, int(2 * self.fduration * self.srate), self.srate,
                                           om_w=self.fbank_config[0],
                                           alp=self.fbank_config[1], fixed=1, bet=self.fbank_config[2], warp_fact=1))
            self.fbank.requires_grad = True
        else:
            if self.complex_modulation:
                self.fbank = self.initialize_filterbank(self.n_filters, int(1 * self.fduration * self.srate),
                                                        self.srate,
                                                        om_w=self.fbank_config[0],
                                                        alp=self.fbank_config[1], fixed=1, bet=self.fbank_config[2],
                                                        warp_fact=1, make_symmetric=True)
            else:
                self.fbank = self.initialize_filterbank(self.n_filters, int(2 * self.fduration * self.srate),
                                                        self.srate,
                                                        om_w=self.fbank_config[0],
                                                        alp=self.fbank_config[1], fixed=1, bet=self.fbank_config[2],
                                                        warp_fact=1)

        # Initialize lifter if needed
        if initialize_lifter:
            lifter = pkl.load(open(initialize_lifter, 'rb'))
            logging.info('Initializing lifter from file {:s}'.format(initialize_lifter))
        else:
            if self.update_lifter_multiband:
                if self.use_complex_lifter:
                    lifter = np.ones((self.n_filters, coeff_num, 2))
                    lifter[:, :, 1] = 0
                else:
                    lifter = np.ones((self.n_filters, coeff_num))
            else:
                if self.use_complex_lifter:
                    lifter = np.ones(coeff_num, 2)
                    lifter[:, 1] = 0
                else:
                    lifter = np.ones(coeff_num)

        lifter /= boost_lifter_lr

        # Convert lifters to tensors
        if self.update_lifter or self.update_lifter_multiband:
            self.lifter = torch.nn.Parameter(torch.tensor(lifter, dtype=self.datatype, device=self.device))
            self.lifter.requires_grad = True
            if self.scale_lifter_gradient:
                # Scale the gradient for lifter
                logging.info('Scaling lifter gradient by {:f}'.format(self.scale_lifter_gradient))
                h = self.lifter.register_hook(lambda grad: grad * self.scale_lifter_gradient)
        else:
            self.lifter = torch.tensor(lifter, dtype=self.datatype, device=self.device)
            # self.lifter = torch.tensor(lifter, dtype=self.datatype)

        self.updatable_params = self.lifter
        if self.freeze_lifter_finetune_updates:  # We start with no updates of the lifter
            if self.update_lifter or self.update_lifter_multiband:
                logging.info('WILL FREEZE LIFTER WEIGHTS AFTER {:d} STEPS'.format(
                    self.freeze_lifter_finetune_updates))
                self.lifter.requires_grad = True

        self.feature_batch = feature_batch
        if spectral_substraction_vector is not None:
            # Loading spectral substration vector
            self.spectral_substraction_vector = torch.tensor(pkl.load(open(spectral_substraction_vector, 'rb')))
            self.spectral_substraction_vector[0] = 0 + 1j * torch.imag(self.spectral_substraction_vector[0])
        else:
            self.spectral_substraction_vector = None

        self.online_normalize = online_normalize

    def dct_type2(self, input: torch.Tensor) -> torch.Tensor:
        """
        Comput Discrete Cosine Transform (DCT-II) on the last tensor dimension

        """

        input_shape = input.shape
        N = input_shape[-1]
        input = input.contiguous().view(-1, N)

        input = torch.cat([input, input.flip([1])], dim=1)
        input = torch.fft.fft(input, dim=1)
        input = input[:, :N]

        k = - torch.arange(N, device=input.device)[None, :] * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)
        input = input.real * W_r - input.imag * W_i

        input = input.view(*input_shape)

        return input

    def compute_autocorr(self, input):
        """
        Compute autocorrelation of input
        """
        input = torch.fft.fft(input, dim=-1)
        if self.complex_modulation:
            input = torch.fft.ifft(input * torch.conj(input), dim=-1)
        else:
            input = torch.fft.ifft(input * torch.conj(input), dim=-1).real
        return input

    def levinson_durbin(self, R, p):
        """
        Levinson Durbin recursion to compute LPC coefficients

        :param R - autocorrelation coefficients: Tensor (batch x num_frames x n_filters x autocorr)
        :param p - lpc model order: int
        :return: Tensor (batch x num_frames x n_filters x lpc_coeff), Tensor (batch x num_frames x n_filters)

        """
        num_batch = R.shape[0]
        num_frames = R.shape[1]
        n_filters = R.shape[2]

        k = torch.zeros((num_batch, num_frames, n_filters, p), dtype=R.dtype, device=R.device)
        alphs = torch.zeros((num_batch, num_frames, n_filters, p, p), dtype=R.dtype, device=R.device)
        errs = torch.zeros((num_batch, num_frames, n_filters, p + 1), dtype=R.dtype, device=R.device)
        errs[:, :, :, 0] = R[:, :, :, 0]

        for i in range(1, p + 1):
            if i == 1:
                k[:, :, :, i - 1] = R[:, :, :, i] / errs[:, :, :, i - 1]
            else:
                k[:, :, :, i - 1] = (R[:, :, :, i] - torch.sum(
                    alphs[:, :, :, 0:i - 1, i - 2] * torch.flip(R[:, :, :, 1:i], [3]), dim=3)) / errs[:, :, :,
                                                                                                 i - 1]
            alphs[:, :, :, i - 1, i - 1] = k[:, :, :, i - 1]
            if i > 1:
                for j in range(1, i):
                    alphs[:, :, :, j - 1, i - 1] = alphs[:, :, :, j - 1, i - 2] - k[:, :, :, i - 1] * torch.conj(
                        alphs[:, :, :,
                        i - j - 1,
                        i - 2])
            errs[:, :, :, i] = (1 - torch.abs(k[:, :, :, i - 1]) ** 2) * errs[:, :, :, i - 1]

        return torch.cat((torch.ones((num_batch, num_frames, n_filters, 1), dtype=R.dtype, device=R.device),
                          -alphs[:, :, :, :, p - 1]), axis=3), errs[:, :, :, -1]

    def compute_lpc(self, input: torch.Tensor, order: int):

        """

        :param input: Tensor (batch x num_frames x n_filters x frame_dim)
        :return: Tensor (batch x num_frames x n_filters x lpc_coeff), Tensor (batch x num_frames x n_filters)
        """

        if self.precision_lpc:
            if self.complex_modulation:
                input = input.to(dtype=torch.complex128)
            else:
                input = input.to(dtype=torch.double)
        R = self.compute_autocorr(input)
        lpc_coeff, gain = self.levinson_durbin(R, p=order)
        if self.precision_lpc:
            if self.complex_modulation:
                lpc_coeff = lpc_coeff.to(dtype=torch.complex64)
                gain = gain.to(dtype=torch.complex64)
            else:
                lpc_coeff = lpc_coeff.to(dtype=torch.float)
                gain = gain.to(dtype=torch.float)

        return lpc_coeff, gain

    def bwe_lpc_stabilizer(self, input: torch.Tensor):
        """

        """
        size = list(input.shape[:-1])
        size.append(1)
        wts = torch.arange(0, self.order + 1).repeat(size)
        gamma = torch.log(torch.ones(input.shape, dtype=input.dtype, device=input.device) * self.bwe_factor)
        wts = torch.exp(gamma * wts)

        for i in range(self.bwe_iter_num):
            input = input * wts
        return input

    def compute_modspec_from_lpc(self, gain, lpc_coeff, lim):
        """
        :param gain: Tensor (batch x  num_frames x n_filters)
        :param lpc_coeff: Tensor (batch x num_frames x n_filters x lpc_num)
        :param lim: int
        :return: Tensor (batch x num_frames x n_filters x num_modspec),
        """

        num_batch = lpc_coeff.shape[0]
        num_frames = lpc_coeff.shape[1]
        n_filters = lpc_coeff.shape[2]
        lpc_coeff[:, :, :, 1:] = -lpc_coeff[:, :, :, 1:]
        lpc_cep = torch.zeros(num_batch, num_frames, n_filters, lim, dtype=lpc_coeff.dtype, device=lpc_coeff.device)

        lpc_cep[:, :, :, 0] = torch.log(torch.sqrt(gain))
        lpc_cep[:, :, :, 1] = lpc_coeff[:, :, :, 1]
        if lpc_coeff.shape[3] < lim:
            lpc_coeff = torch.cat(
                [lpc_coeff, torch.zeros(num_batch, num_frames, n_filters, int(lim - lpc_coeff.shape[3] + 1),
                                        device=lpc_coeff.device)], axis=3)
        for n in range(2, lim):
            a = torch.arange(1, n) / n
            a = a.to(lpc_coeff.device)
            b = torch.flip(lpc_coeff[:, :, :, 1:n], dims=[3])
            c = lpc_cep[:, :, :, 1:n]
            acc = torch.sum(a * b * c, axis=3)
            lpc_cep[:, :, :, n] = acc + lpc_coeff[:, :, :, n]
        return lpc_cep

    def get_frames(self, signal: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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

        if self.feature_batch is not None:
            # Reshape to have longer utterances, helps in feature extraction
            # Might not be equally divisible, deal with that
            sig_size = signal.shape[0] * signal.shape[1]
            div_req = self.feature_batch
            div_reminder = sig_size % div_req
            signal = signal.flatten()
            if div_reminder != 0:
                if div_reminder < int(div_req / 2):
                    signal = signal[:-div_reminder]
                else:
                    signal = torch.cat([signal, torch.zeros(div_req - div_reminder, device=signal.device)])
            signal = torch.reshape(signal, (self.feature_batch, -1))

        tsamples = signal.shape[1]

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
        return tsamples, frames

    def get_normalizing_vector_old(self, signal, fduration, overlap_fraction, append_len=500000, discont=np.pi,
                                   no_window=True, phase_max_cap=200):

        overlap_fraction = 1 - overlap_fraction
        lfr = 1 / (overlap_fraction * fduration)
        flength_samples = int(self.srate * fduration)
        frate_samples = int(self.srate / lfr)

        if flength_samples % 2 == 0:
            sp_b = int(flength_samples / 2) - 1
            sp_f = int(flength_samples / 2)
            extend = int(flength_samples / 2) - 1
        else:
            sp_b = int((flength_samples - 1) / 2)
            sp_f = int((flength_samples - 1) / 2)
            extend = int((flength_samples - 1) / 2)

        if self.feature_batch is not None:
            # Reshape to have longer utterances, helps in feature extraction
            # Might not be equally divisible, deal with that
            sig_size = signal.shape[0] * signal.shape[1]
            div_req = self.feature_batch
            div_reminder = sig_size % div_req
            signal = signal.flatten()
            if div_reminder != 0:
                if div_reminder < int(div_req / 2):
                    signal = signal[:-div_reminder]
                else:
                    signal = torch.cat([signal, torch.zeros(div_req - div_reminder, device=signal.device)])
            signal = torch.reshape(signal, (self.feature_batch, -1))

        tsamples = signal.shape[1]

        # signal = torch.nn.functional.pad(signal.unsqueeze(1), (extend, extend), mode='constant', value=0.0).squeeze(1)
        signal = torch.nn.functional.pad(signal.unsqueeze(1), (extend, extend), mode='constant', value=0.0).squeeze(1)

        signal_length = signal.shape[1]

        win = torch.hamming_window(flength_samples, dtype=signal.dtype, device=signal.device)

        idx = sp_b
        frames = []
        while (idx + sp_f) < signal_length:
            if no_window:
                frames.append(signal[:, idx - sp_b:idx + sp_f + 1].unsqueeze(1))
            else:
                frames.append(signal[:, idx - sp_b:idx + sp_f + 1].unsqueeze(1) * win)
            idx += frate_samples

        frames = torch.cat(frames, dim=1)

        frames = torch.cat([frames, torch.zeros(frames.shape[0], frames.shape[1], append_len - frames.shape[2],
                                                dtype=frames.dtype, device=frames.device)], dim=-1)
        frames = frames[:, :, 0:append_len]
        frames = torch.log(torch.fft.fft(frames, axis=-1))
        frames = torch.reshape(frames, (frames.shape[0] * frames.shape[1], -1))

        total_num_frames = frames.shape[0]
        phase = self.phase_unwrap(torch.imag(frames), discont=discont)
        logmag = torch.real(frames)

        phase = torch.sum(phase, dim=0) / total_num_frames
        logmag = torch.sum(logmag, dim=0) / total_num_frames

        ## Adjust the phase
        phi = (phase[-1] - phase[0]) / phase.shape[0]
        x_ph = torch.arange(phase.shape[0], device=frames.device)
        y_ph = phase[0] + x_ph * phi
        ph_corrected = y_ph - phase
        ph_corrected = ph_corrected * phase_max_cap / torch.max(ph_corrected)

        ssv = logmag + 1j * ph_corrected

        return logmag, phase, ph_corrected, ssv

    def get_normalizing_vector(self, signal, fduration, overlap_fraction, append_len=500000, discont=np.pi,
                               no_window=True, phase_max_cap=200):

        overlap_fraction = 1 - overlap_fraction
        lfr = 1 / (overlap_fraction * fduration)
        flength_samples = int(self.srate * fduration)
        frate_samples = int(self.srate / lfr)

        if flength_samples % 2 == 0:
            sp_b = int(flength_samples / 2) - 1
            sp_f = int(flength_samples / 2)
            extend = int(flength_samples / 2) - 1
        else:
            sp_b = int((flength_samples - 1) / 2)
            sp_f = int((flength_samples - 1) / 2)
            extend = int((flength_samples - 1) / 2)

        if self.feature_batch is not None:
            # Reshape to have longer utterances, helps in feature extraction
            # Might not be equally divisible, deal with that
            sig_size = signal.shape[0] * signal.shape[1]
            div_req = self.feature_batch
            div_reminder = sig_size % div_req
            signal = signal.flatten()
            if div_reminder != 0:
                if div_reminder < int(div_req / 2):
                    signal = signal[:-div_reminder]
                else:
                    signal = torch.cat([signal, torch.zeros(div_req - div_reminder, device=signal.device)])
            signal = torch.reshape(signal, (self.feature_batch, -1))

        tsamples = signal.shape[1]

        # signal = torch.nn.functional.pad(signal.unsqueeze(1), (extend, extend), mode='constant', value=0.0).squeeze(1)
        signal = torch.nn.functional.pad(signal.unsqueeze(1), (extend, extend), mode='constant', value=0.0).squeeze(1)

        signal_length = signal.shape[1]

        win = torch.hamming_window(flength_samples, dtype=signal.dtype, device=signal.device)

        idx = sp_b
        frames = []
        while (idx + sp_f) < signal_length:
            if no_window:
                frames.append(signal[:, idx - sp_b:idx + sp_f + 1].unsqueeze(1))
            else:
                frames.append(signal[:, idx - sp_b:idx + sp_f + 1].unsqueeze(1) * win)
            idx += frate_samples

        frames = torch.cat(frames, dim=1)

        frames = torch.cat([frames, torch.zeros(frames.shape[0], frames.shape[1], append_len - frames.shape[2],
                                                dtype=frames.dtype, device=frames.device)], dim=-1)
        frames = frames[:, :, 0:append_len]
        frames = torch.log(torch.fft.fft(frames, axis=-1))  # num_batch x num_frames x dimension
        # frames = torch.reshape(frames, (frames.shape[0] * frames.shape[1], -1))

        total_num_frames = frames.shape[1]
        phase = self.phase_unwrap(torch.imag(frames), discont=discont)  # num_batch x num_frames x dimension
        logmag = torch.real(frames)  # num_batch x num_frames x dimension

        phase = torch.sum(phase, dim=1) / total_num_frames  # num_batch x dimension
        logmag = torch.sum(logmag, dim=1) / total_num_frames  # num_batch x dimension

        ## Adjust the phase
        batch_num = logmag.shape[0]
        dim = logmag.shape[1]
        ph_corrected = torch.zeros(batch_num, dim, device=phase.device)
        for i in range(batch_num):
            phi = (phase[i, -1] - phase[i, 0]) / dim
            x_ph = torch.arange(dim, device=frames.device)
            y_ph = phase[i, 0] + x_ph * phi
            ph_corrected[i, :] = y_ph - phase
            ph_corrected[i, :] = ph_corrected[i, :] * phase_max_cap / torch.max(ph_corrected[i, :])

        ssv = logmag + 1j * ph_corrected

        return logmag, phase, ph_corrected, ssv

    def phase_unwrap(self, phase, discont=np.pi):

        phase_numpy = phase.cpu().detach().numpy()
        phase_numpy = np.unwrap(phase_numpy, discont=discont, axis=-1)
        phase_numpy = torch.from_numpy(phase_numpy)
        phase_numpy = phase_numpy.to(phase.dtype)
        phase = phase_numpy.to(phase.device)

        return phase

    def __warp_func_bark(self, x, warp_fact=1):
        import numpy as np
        return 6 * np.arcsinh((x / warp_fact) / 600)

    def initialize_filterbank(self, nfilters, nfft, srate, om_w=1, alp=1, fixed=1, bet=2.5, warp_fact=1,
                              make_symmetric=False):
        f_max = srate / 2
        warped_max = self.__warp_func_bark(f_max, warp_fact)
        fwarped_cf = np.linspace(0, warped_max, nfilters)
        f_linear = np.linspace(0, f_max, int(np.floor(nfft / 2 + 1)))
        f_warped = self.__warp_func_bark(f_linear, warp_fact)
        filts = np.zeros((nfilters, int(np.floor(nfft / 2 + 1))))
        alp_c = alp
        for i in range(nfilters):
            fc = fwarped_cf[i]
            if fixed == 1:
                alp = alp_c
            else:
                alp = alp_c * np.exp(-0.1 * fc)
            for j, fw in enumerate(f_warped):
                if fw - fc <= -om_w / 2:
                    filts[i, j] = np.power(10, alp * (fw - fc + om_w / 2))
                elif fw - fc > -om_w / 2 and fw - fc < om_w / 2:
                    filts[i, j] = 1
                else:
                    filts[i, j] = np.power(10, -bet * (fw - fc - om_w / 2))
        # return torch.tensor(filts, dtype=self.datatype)
        if make_symmetric:
            filts = np.concatenate((filts[:, :-1], np.flip(filts, axis=1)), axis=1)

        return torch.tensor(filts, dtype=self.datatype, device=self.device)

    def OLA(self, modspec, t_samples, dtype, device):

        num_batch = modspec.shape[0]
        num_frames = modspec.shape[1]
        feats = torch.zeros((num_batch, int(np.ceil(t_samples * self.frate / self.srate)), self.n_filters), dtype=dtype,
                            device=device)
        ptr = int(0)
        ### Overlap and Add stage
        for j in range(0, num_frames):
            if j == 0:
                if feats.shape[1] < self.cut_half:
                    feats += modspec[:, j, :self.cut_half:self.cut_half + feats.shape[1], :]
                else:
                    feats[:, ptr:ptr + self.cut_half, :] += modspec[:, j, self.cut_half:, :]

            elif j == num_frames - 1 or j == num_frames - 2:
                if modspec.shape[2] >= feats.shape[1] - ptr:
                    feats[:, ptr:, :] += modspec[:, j, :feats.shape[1] - ptr, :]
                else:
                    feats[:, ptr:ptr + self.cut, :] += modspec[:, j, :, :]
            else:
                feats[:, ptr:ptr + self.cut, :] += modspec[:, j, :, :]

            if j == 0:
                ptr = int(ptr + self.cut_overlap - self.cut_half)
            else:
                # ptr = int(ptr + self.cut_overlap + randrange(2))
                ptr = int(ptr + self.cut_overlap + 1)

        feats = torch.log(torch.clip(feats, max=None, min=0.0000001))
        feats = torch.nan_to_num(feats, nan=0.0000001, posinf=0.0000001, neginf=0.0000001)  # Probably not the best idea

        return feats

    def compute_spectrogram(self, input: torch.Tensor, ilens: torch.Tensor = None) -> Tuple[
        torch.Tensor, Optional[torch.Tensor]]:
        """Compute FDLP-Spectrogram With Matrices.

        Args:
            input: (Batch, Nsamples) or (Batch, Nsample)
            ilens: (Batch)
        Returns:
            output: (Batch, Frames, Freq) or (Batch, Frames, Freq)

        """

        if self.online_normalize:
            _, _, _, self.spectral_substraction_vector = self.get_normalizing_vector(input, fduration=25,
                                                                                     overlap_fraction=0.98,
                                                                                     append_len=500000, discont=np.pi)

        num_batch = input.shape[0]
        # First divide the signal into frames

        if self.spectral_substraction_vector is not None and self.dereverb_whole_sentence:
            input = self.dereverb_whole(input, self.spectral_substraction_vector)

        t_samples, frames = self.get_frames(input)
        num_frames = frames.shape[1]

        if self.spectral_substraction_vector is not None and not self.dereverb_whole_sentence:
            self.spectral_substraction_vector = self.spectral_substraction_vector.to(input.device)
            # logging.info('Substracting spectral vector')
            frames = self.spectral_substraction_preprocessing(frames)

        # Compute DCT (olens remains the same)
        if self.complex_modulation:
            frames = torch.fft.ifft(frames) * int(self.srate * self.fduration)
        else:
            frames = self.dct_type2(frames) / np.sqrt(2 * int(self.srate * self.fduration))

        # Put fbank, mask and lifter into proper device if they are already not there
        if self.fbank.device.type != input.device.type:
            print('Transferring fbank, mask and lifter to {:s}'.format(input.device.type))
            self.fbank = self.fbank.to(input.device)
            self.lifter = self.lifter.to(input.device)
            self.mask = self.mask.to(input.device)

        # Divide into sub-bands
        frames = frames.unsqueeze(2).repeat(1, 1, self.n_filters, 1)
        frames = frames * self.fbank[:, 0:-1]  # batch x num_frames x n_filters x frame_dim of 1.5 secs

        han_weight = torch.hann_window(self.cut, dtype=input.dtype, device=input.device)
        ham_weight = torch.hamming_window(self.cut, dtype=input.dtype, device=input.device)

        if self.num_chunks:

            # Divide computation to num_chunks along num_batch direction
            chunk_size = int(np.ceil(frames.shape[0] / self.num_chunks))
            frames = list(torch.split(frames, split_size_or_sections=chunk_size, dim=0))

            for chunk_idx in range(len(frames)):
                frames[chunk_idx], gain = self.compute_lpc(frames[chunk_idx],
                                                           self.order)  # batch x num_frames x n_filters x lpc_coeff
                if self.do_bwe:
                    frames[chunk_idx] = self.bwe_lpc_stabilizer(frames[chunk_idx])

                frames[chunk_idx] = self.compute_modspec_from_lpc(gain, frames[chunk_idx],
                                                                  self.coeff_num)  # batch x num_frames x n_filters x num_modspec

            frames = torch.cat(frames, dim=0)

        else:

            ### Compute all LPC in all bands and all frames parallely to make it fast

            frames, gain = self.compute_lpc(frames, self.order)  # batch x num_frames x n_filters x lpc_coeff
            if self.do_bwe:
                frames = self.bwe_lpc_stabilizer(frames)

            frames = self.compute_modspec_from_lpc(gain, frames,
                                                   self.coeff_num)  # batch x num_frames x n_filters x num_modspec
        modspec = frames

        modspec = modspec * self.mask  # (batch x num_frames x n_filters x num_modspec)
        # logging.info('Boost rate {}'.format(self.boost_lifter_lr.data))
        # logging.info('lifter mean'.format(torch.mean(self.lifter).data))
        sys.stdout.flush()
        if self.update_lifter_multiband:
            if self.lnlt:
                modspec = modspec * self.lnlt(
                    self.boost_lifter_lr * self.lifter.unsqueeze(0).unsqueeze(0).repeat(num_batch, num_frames, 1,
                                                                                        1))  # (batch x num_frames x n_filters x num_modspec)
            else:
                if self.use_complex_lifter:

                    modspec = modspec * torch.view_as_complex(self.boost_lifter_lr * self.lifter).unsqueeze(
                        0).unsqueeze(0).repeat(num_batch,
                                               num_frames,
                                               1,
                                               1)  # (batch x num_frames x n_filters x num_modspec)
                else:
                    modspec = modspec * self.boost_lifter_lr * self.lifter.unsqueeze(0).unsqueeze(0).repeat(num_batch,
                                                                                                            num_frames,
                                                                                                            1,
                                                                                                            1)  # (batch x num_frames x n_filters x num_modspec)
        else:
            if self.lnlt:
                modspec = modspec * self.boost_lifter_lr * self.lnlt(
                    self.lifter)  # (batch x num_frames x n_filters x num_modspec)
            else:
                modspec = modspec * self.boost_lifter_lr * self.lifter  # (batch x num_frames x n_filters x num_modspec)
        if self.complex_modulation:
            modspec = torch.fft.fft(modspec, 1 * int(round(
                self.fduration * self.frate)))  # (batch x num_frames x n_filters x int(self.fduration * self.frate))
        else:
            modspec = torch.fft.fft(modspec, 2 * int(round(
                self.fduration * self.frate)))  # (batch x num_frames x n_filters x int(self.fduration * self.frate))
        modspec = torch.abs(torch.exp(modspec))
        modspec = modspec[:, :, :, 0:self.cut] * han_weight / ham_weight
        modspec = torch.transpose(modspec, 2, 3)  # (batch x num_frames x int(self.fduration * self.frate) x n_filters)

        # OVERLAP AND ADD

        modspec = self.OLA(modspec=modspec, t_samples=t_samples, dtype=input.dtype, device=input.device)

        if self.feature_batch is not None:
            # Might not be equally divisible, deal with that
            modspec_size = modspec.shape[0] * modspec.shape[1] * modspec.shape[2]
            div_req = num_batch * self.n_filters
            div_reminder = modspec_size % div_req
            if div_reminder != 0:
                modspec = modspec.flatten()
                if div_reminder < int(div_req / 2):
                    modspec = modspec[:-div_reminder]
                else:
                    modspec = torch.cat([modspec, torch.zeros(div_req - div_reminder, device=input.device)])

            modspec = torch.reshape(modspec, (num_batch, -1, self.n_filters))

        if ilens is not None:
            olens = torch.floor(ilens * self.frate / self.srate)
            olens = olens.to(ilens.dtype)
            modspec.masked_fill_(make_pad_mask(olens, modspec, 1), 0.0000001)
            modspec = modspec[:, :torch.max(olens), :]
        else:
            olens = None

        return modspec, olens

    def dereverb_whole(self, signal, rir_mag):
        sig_shape = signal.shape[1]

        # Make noise the same shape as speech
        if rir_mag.shape[0] > signal.shape[1]:
            signal = torch.cat(
                (signal, torch.zeros(signal.shape[0], rir_mag.shape[0] - signal.shape[1], device=signal.device,
                                     dtype=signal.dtype)), dim=-1)
        else:
            signal = signal[:, 0:rir_mag.shape[0]]
        signal = torch.real(torch.fft.ifft(torch.exp(torch.log(torch.fft.fft(signal)) - rir_mag))).type(torch.float32)

        return signal[:, :sig_shape]

    def spectral_substraction_preprocessing(self, frames):

        ori_len = frames.shape[-1]

        if self.spectral_substraction_vector.shape[0] > frames.shape[-1]:

            frames = torch.cat((frames, torch.zeros(
                frames.shape[0], frames.shape[1], self.spectral_substraction_vector.shape[0] - frames.shape[-1],
                device=frames.device)), dim=-1)
        else:
            frames = frames[:, :0:self.spectral_substraction_vector.shape[0]]

        frames_fft = torch.log(torch.fft.fft(frames))
        frames_fft = torch.real(torch.fft.ifft(torch.exp(frames_fft - self.spectral_substraction_vector)))

        return frames_fft[:, :, :ori_len]

    def forward(self, input: torch.Tensor, ilens: torch.Tensor = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute FDLP-Spectrogram forward function.

        Args:
            input: (Batch, Nsamples) or (Batch, Nsample, Channels)
            ilens: (Batch)
        Returns:
            output: (Batch, Frames, Freq) or (Batch, Frames, Channels, Freq)
        """

        bs = input.size(0)

        if self.freeze_lifter_finetune_updates:
            ft = self.freeze_lifter_finetune_updates <= self.num_updates  # Fine tune after ft is True
            if self.num_updates <= self.freeze_lifter_finetune_updates:
                self.num_updates += 1

            if ft and self.lifter.requires_grad is True:
                logging.info('STOP LIFTER WEIGHT UPDATES FROM THIS STEP')
                self.lifter.requires_grad = False  # Turn off lifter updates

        if input.dim() == 3:
            multi_channel = True
            # input: (Batch, Nsample, Channels) -> (Batch * Channels, Nsample)
            input = input.transpose(1, 2).reshape(-1, input.size(1))
        else:
            multi_channel = False
        output, olens = self.compute_spectrogram(input, ilens)
        if multi_channel:
            # output: (Batch * Channel, Frames, Freq, 2=real_imag)
            # -> (Batch, Frame, Channel, Freq, 2=real_imag)
            output = output.view(bs, -1, output.size(1), output.size(2)).transpose(
                1, 2
            )

        return output, olens


class fdlp_spectrogram_dereverb(fdlp_spectrogram):
    def __init__(self,
                 dereverb_num_layers: int = 1,
                 dereverb_num_units: int = 765,
                 **kwargs
                 ):
        assert check_argument_types()
        super().__init__(**kwargs)

        logging.info(
            'Setting up a dereverberation bLSTM with {:d} layer and {:d} hidden units'.format(dereverb_num_layers,
                                                                                              dereverb_num_units))
        self.dereverb_num_layers = dereverb_num_layers
        self.dereverb_num_units = dereverb_num_units
        # self.dereverb_lstm = torch.nn.LSTM(input_size=int(self.fduration * self.srate),
        #                                   hidden_size=self.dereverb_num_units, num_layers=self.dereverb_num_layers,
        #                                   bidirectional=True, batch_first=True)
        # self.real_projection = torch.nn.Linear(in_features=2 * self.dereverb_num_units,
        #                                       out_features=int(self.fduration * self.srate / 2))
        # self.imag_projection = torch.nn.Linear(in_features=2 * self.dereverb_num_units,
        #                                       out_features=int(self.fduration * self.srate / 2))

        self.dereverb_lstm = torch.nn.LSTM(input_size=2 * self.n_filters * self.coeff_num,
                                           hidden_size=self.dereverb_num_units, num_layers=self.dereverb_num_layers,
                                           bidirectional=True, batch_first=True)
        self.real_projection = torch.nn.Linear(in_features=2 * self.dereverb_num_units,
                                               out_features=self.n_filters * self.coeff_num)
        self.imag_projection = torch.nn.Linear(in_features=2 * self.dereverb_num_units,
                                               out_features=self.n_filters * self.coeff_num)

    def dereverb(self, frames):
        # torch.autograd.set_detect_anomaly(True)
        frames = torch.fft.fft(frames)  # Batch x num_frames x frequencies i.e. int(self.fduration * self.srate)
        frames_half = torch.reshape(torch.view_as_real(frames[:, :, :int(self.fduration * self.srate / 2)]), (
            frames.shape[0], frames.shape[1], -1))  # Batch x num_frames x int(self.fduration * self.srate)

        frames_dereverb = self.dereverb_lstm(frames_half)[0]  # Pass through blstm layers
        frames_dereverb = [self.real_projection(frames_dereverb),
                           self.imag_projection(frames_dereverb)]  # Project onto proper dimension
        frames_dereverb = [torch.mean(frames_dereverb[0], dim=1),
                           torch.mean(frames_dereverb[1], dim=1)]  # Compute sequence summary
        frames_dereverb = [torch.cat([frames_dereverb[0], torch.flip(frames_dereverb[0], dims=[1])], dim=1),
                           torch.cat([frames_dereverb[1], -torch.flip(frames_dereverb[1], dims=[1])],
                                     dim=1)]  # Compute full spectrum from half spectrum

        frames_dereverb = torch.view_as_complex(
            torch.cat([frames_dereverb[0].unsqueeze(-1), frames_dereverb[1].unsqueeze(-1)], dim=-1))
        frames_dereverb = frames_dereverb.unsqueeze(1)
        return torch.real(torch.fft.ifft(frames * frames_dereverb))

    def dereverb_modspec(self, frames):
        # torch.autograd.set_detect_anomaly(True)

        frames = torch.reshape(frames,
                               (frames.shape[0], frames.shape[1], -1))  # (batch x num_frames x n_filters * num_modspec)
        # frames = torch.log(torch.fft.ifft(torch.exp(torch.fft.fft(frames))))
        frames = torch.log(frames)

        frames_real = torch.reshape(torch.view_as_real(frames), (
            frames.shape[0], frames.shape[1], -1))  # (batch x num_frames x 2*n_filters * num_modspec)

        frames_dereverb = self.dereverb_lstm(frames_real)[0]  # Pass through blstm layers

        frames_dereverb = [self.real_projection(frames_dereverb),  # (batch x num_frames x n_filters * num_modspec)
                           self.imag_projection(frames_dereverb)]  # Project onto proper dimension

        frames_dereverb = [torch.mean(frames_dereverb[0], dim=1),  # (batch x n_filters * num_modspec)
                           torch.mean(frames_dereverb[1], dim=1)]  # Compute sequence summary

        # frames_dereverb = [torch.mean(frames_dereverb[0], dim=0),  # ( batch x n_filters * num_modspec)
        #                   torch.mean(frames_dereverb[1], dim=0)]  # Compute sequence summary

        frames_dereverb = torch.view_as_complex(
            torch.cat([frames_dereverb[0].unsqueeze(-1), frames_dereverb[1].unsqueeze(-1)],
                      dim=-1))  # (batch x n_filters * num_modspec)

        frames_dereverb = frames_dereverb.unsqueeze(1)  # ( batch x 1 x n_filters * num_modspec)
        # frames_dereverb = frames_dereverb.unsqueeze(0)
        frames = frames - frames_dereverb
        # frames = frames - frames_dereverb
        # frames = torch.fft.ifft(torch.log(torch.fft.fft(torch.exp(frames))))
        frames = torch.exp(frames)
        return torch.reshape(frames, (frames.shape[0], frames.shape[1], self.n_filters, self.coeff_num))

    def compute_spectrogram(self, input: torch.Tensor, ilens: torch.Tensor = None) -> Tuple[
        torch.Tensor, Optional[torch.Tensor]]:
        """Compute FDLP-Spectrogram With Matrices.

        Args:
            input: (Batch, Nsamples) or (Batch, Nsample)
            ilens: (Batch)
        Returns:
            output: (Batch, Frames, Freq) or (Batch, Frames, Freq)

        """

        t_samples = input.shape[1]
        num_batch = input.shape[0]

        # First divide the signal into frames

        frames = self.get_frames(input)
        # frames = self.dereverb(frames)
        num_frames = frames.shape[1]

        # Compute DCT (olens remains the same)
        if self.complex_modulation:
            frames = torch.fft.ifft(frames) * int(self.srate * self.fduration)
        else:
            frames = self.dct_type2(frames) / np.sqrt(2 * int(self.srate * self.fduration))

        # Put fbank, mask and lifter into proper device if they are already not there
        if self.fbank.device.type != input.device.type:
            print('Transferring fbank, mask and lifter to {:s}'.format(input.device.type))
            self.fbank = self.fbank.to(input.device)
            self.lifter = self.lifter.to(input.device)
            self.mask = self.mask.to(input.device)

        # Divide into sub-bands
        frames = frames.unsqueeze(2).repeat(1, 1, self.n_filters, 1)
        frames = frames * self.fbank[:, 0:-1]  # batch x num_frames x n_filters x frame_dim of 1.5 secs

        han_weight = torch.hann_window(self.cut, dtype=input.dtype, device=input.device)
        ham_weight = torch.hamming_window(self.cut, dtype=input.dtype, device=input.device)

        if self.num_chunks:

            # Divide computation to num_chunks along num_batch direction
            chunk_size = int(np.ceil(frames.shape[0] / self.num_chunks))
            frames = list(torch.split(frames, split_size_or_sections=chunk_size, dim=0))

            for chunk_idx in range(len(frames)):
                frames[chunk_idx], gain = self.compute_lpc(frames[chunk_idx],
                                                           self.order)  # batch x num_frames x n_filters x lpc_coeff
                if self.do_bwe:
                    frames[chunk_idx] = self.bwe_lpc_stabilizer(frames[chunk_idx])

                frames[chunk_idx] = self.compute_modspec_from_lpc(gain, frames[chunk_idx],
                                                                  self.coeff_num)  # batch x num_frames x n_filters x num_modspec

            frames = torch.cat(frames, dim=0)

        else:

            ### Compute all LPC in all bands and all frames parallely to make it fast

            frames, gain = self.compute_lpc(frames, self.order)  # batch x num_frames x n_filters x lpc_coeff
            if self.do_bwe:
                frames = self.bwe_lpc_stabilizer(frames)

            frames = self.compute_modspec_from_lpc(gain, frames,
                                                   self.coeff_num)  # batch x num_frames x n_filters x num_modspec
        modspec = frames
        modspec = self.dereverb_modspec(modspec)
        modspec = modspec * self.mask  # (batch x num_frames x n_filters x num_modspec)
        # logging.info('Boost rate {}'.format(self.boost_lifter_lr.data))
        # logging.info('lifter mean'.format(torch.mean(self.lifter).data))
        sys.stdout.flush()
        if self.update_lifter_multiband:
            if self.lnlt:
                modspec = modspec * self.lnlt(
                    self.boost_lifter_lr * self.lifter.unsqueeze(0).unsqueeze(0).repeat(num_batch, num_frames, 1,
                                                                                        1))  # (batch x num_frames x n_filters x num_modspec)
            else:
                if self.use_complex_lifter:

                    modspec = modspec * torch.view_as_complex(self.boost_lifter_lr * self.lifter).unsqueeze(
                        0).unsqueeze(0).repeat(num_batch,
                                               num_frames,
                                               1,
                                               1)  # (batch x num_frames x n_filters x num_modspec)
                else:
                    modspec = modspec * self.boost_lifter_lr * self.lifter.unsqueeze(0).unsqueeze(0).repeat(
                        num_batch,
                        num_frames,
                        1,
                        1)  # (batch x num_frames x n_filters x num_modspec)
        else:
            if self.lnlt:
                modspec = modspec * self.boost_lifter_lr * self.lnlt(
                    self.lifter)  # (batch x num_frames x n_filters x num_modspec)
            else:
                modspec = modspec * self.boost_lifter_lr * self.lifter  # (batch x num_frames x n_filters x num_modspec)
        if self.complex_modulation:
            modspec = torch.fft.fft(modspec, 1 * int(round(
                self.fduration * self.frate)))  # (batch x num_frames x n_filters x int(self.fduration * self.frate))
        else:
            modspec = torch.fft.fft(modspec, 2 * int(round(
                self.fduration * self.frate)))  # (batch x num_frames x n_filters x int(self.fduration * self.frate))
        modspec = torch.abs(torch.exp(modspec))
        modspec = modspec[:, :, :, 0:self.cut] * han_weight / ham_weight
        modspec = torch.transpose(modspec, 2,
                                  3)  # (batch x num_frames x int(self.fduration * self.frate) x n_filters)

        # OVERLAP AND ADD
        modspec = self.OLA(modspec=modspec, t_samples=t_samples, dtype=input.dtype, device=input.device)

        if ilens is not None:
            olens = torch.floor(ilens * self.frate / self.srate)
            olens = olens.to(ilens.dtype)
            modspec.masked_fill_(make_pad_mask(olens, modspec, 1), 0.0000001)
            modspec = modspec[:, :torch.max(olens), :]
        else:
            olens = None

        return modspec, olens


class fdlp_spectrogram_with_mmh(fdlp_spectrogram):
    def __init__(self,
                 num_modulation_head: int = 1,
                 update_mmh: bool = True,
                 log_scaled_modulation_filters: bool = True,
                 stop_mmh_updates_after: int = None,
                 **kwargs
                 ):
        assert check_argument_types()
        super().__init__(**kwargs)

        self.num_modulation_head = num_modulation_head
        self.update_mmh = update_mmh
        self.stop_mmh_updates_after = stop_mmh_updates_after
        self.log_scaled_modulation_filters = log_scaled_modulation_filters
        self.mod_freq_max = self.coeff_num / self.fduration
        mod_freq_ranges = np.ones((self.n_filters, num_modulation_head + 1))

        if self.log_scaled_modulation_filters:
            mod_freq_ranges[:, :] = np.exp(
                np.linspace(0, np.log(self.mod_freq_max), num_modulation_head + 1)) / self.mod_freq_max
        else:
            mod_freq_ranges[:, :] = np.linspace(0, self.mod_freq_max, num_modulation_head + 1) / self.mod_freq_max

        if self.update_mmh:
            self.mod_freq_ranges = torch.nn.Parameter(
                torch.tensor(mod_freq_ranges, dtype=self.datatype, device=self.device))
            self.mod_freq_ranges.requires_grad = True
        else:
            self.mod_freq_ranges = torch.tensor(mod_freq_ranges, dtype=self.datatype, device=self.device)
        self.rr = torch.nn.ReLU()
        self.updatable_params = self.mod_freq_ranges

        if not self.update_mmh:
            # self.mod_freq_ranges = self.mod_freq_ranges.to(input.device)
            t = torch.linspace(0, self.fduration, 2 * self.coeff_num)

            # We will have multiple lifters now
            self.lifter = []
            for j in range(self.num_modulation_head):
                lifter = torch.zeros(self.n_filters, self.coeff_num)
                for i in range(self.n_filters):
                    temp = 2 * (self.rr(self.mod_freq_max * self.mod_freq_ranges[i, j + 1]) + self.rr(
                        self.mod_freq_max * self.mod_freq_ranges[i, j + 0])) * torch.sinc(
                        2 * (self.rr(self.mod_freq_max * self.mod_freq_ranges[i, j + 1]) + self.rr(
                            self.mod_freq_max * self.mod_freq_ranges[i, j + 0])) * t) - 2 * self.rr(
                        self.mod_freq_max * self.mod_freq_ranges[i, j + 0]) * torch.sinc(
                        2 * self.rr(self.mod_freq_max * self.mod_freq_ranges[i, j + 0]) * t)

                    lifter[i, :] = torch.real(torch.fft.fft(temp)[0:self.coeff_num] / temp.shape[0])
                    lifter = lifter
                self.lifter.append(lifter)

        if self.stop_mmh_updates_after:
            logging.info('MMH WILL BE UPDATED TILL UPDATE NUMBER {:d}'.format(self.stop_mmh_updates_after))

    def OLA(self, modspec, t_samples, dtype, device):

        num_batch = modspec.shape[0]
        num_frames = modspec.shape[1]
        feats = torch.zeros(
            (num_batch, int(np.ceil(t_samples * self.frate / self.srate)), self.n_filters * self.num_modulation_head),
            dtype=dtype,
            device=device)
        ptr = int(0)
        ### Overlap and Add stage
        for j in range(0, num_frames):
            if j == 0:
                if feats.shape[1] < self.cut_half:
                    feats += modspec[:, j, :self.cut_half:self.cut_half + feats.shape[1], :]
                else:
                    feats[:, ptr:ptr + self.cut_half, :] += modspec[:, j, self.cut_half:, :]

            elif j == num_frames - 1 or j == num_frames - 2:
                if modspec.shape[2] >= feats.shape[1] - ptr:
                    feats[:, ptr:, :] += modspec[:, j, :feats.shape[1] - ptr, :]
                else:
                    feats[:, ptr:ptr + self.cut, :] += modspec[:, j, :, :]
            else:
                feats[:, ptr:ptr + self.cut, :] += modspec[:, j, :, :]

            if j == 0:
                ptr = int(ptr + self.cut_overlap - self.cut_half)
            else:
                # ptr = int(ptr + self.cut_overlap + randrange(2))
                ptr = int(ptr + self.cut_overlap + 1)

        feats = torch.log(torch.clip(feats, max=None, min=0.0000001))
        feats = torch.nan_to_num(feats, nan=0.0000001, posinf=0.0000001, neginf=0.0000001)  # Probably not the best idea

        return feats

    def compute_spectrogram(self, input: torch.Tensor, ilens: torch.Tensor = None) -> Tuple[
        torch.Tensor, Optional[torch.Tensor]]:
        """Compute FDLP-Spectrogram With Matrices.

        Args:
            input: (Batch, Nsamples) or (Batch, Nsample)
            ilens: (Batch)
        Returns:
            output: (Batch, Frames, Freq) or (Batch, Frames, Freq)

        """

        t_samples = input.shape[1]
        num_batch = input.shape[0]

        if self.update_mmh:
            # self.mod_freq_ranges = self.mod_freq_ranges.to(input.device)
            t = torch.linspace(0, self.fduration, 2 * self.coeff_num).to(input.device)

            # We will have multiple lifters now
            self.lifter = []
            for j in range(self.num_modulation_head):
                lifter = torch.zeros(self.n_filters, self.coeff_num)
                for i in range(self.n_filters):
                    temp = 2 * (self.rr(self.mod_freq_max * self.mod_freq_ranges[i, j + 1]) + self.rr(
                        self.mod_freq_max * self.mod_freq_ranges[i, j + 0])) * torch.sinc(
                        2 * (self.rr(self.mod_freq_max * self.mod_freq_ranges[i, j + 1]) + self.rr(
                            self.mod_freq_max * self.mod_freq_ranges[i, j + 0])) * t) - 2 * self.rr(
                        self.mod_freq_max * self.mod_freq_ranges[i, j + 0]) * torch.sinc(
                        2 * self.rr(self.mod_freq_max * self.mod_freq_ranges[i, j + 0]) * t)

                    lifter[i, :] = torch.real(torch.fft.fft(temp)[0:self.coeff_num] / temp.shape[0])
                    lifter = lifter.to(input.device)
                self.lifter.append(lifter)
        elif self.lifter[0].device.type != input.device.type:
            for j in range(self.num_modulation_head):
                self.lifter[j] = self.lifter[j].to(input.device)

        # First divide the signal into frames

        frames = self.get_frames(input)
        num_frames = frames.shape[1]

        # Compute DCT (olens remains the same)
        if self.complex_modulation:
            frames = torch.fft.ifft(frames) * int(self.srate * self.fduration)
        else:
            frames = self.dct_type2(frames) / np.sqrt(2 * int(self.srate * self.fduration))

        # Put fbank, mask and lifter into proper device if they are already not there
        if self.fbank.device.type != input.device.type:
            print('Transferring fbank, mask and lifter to {:s}'.format(input.device.type))
            self.fbank = self.fbank.to(input.device)
            # self.lifter = self.lifter.to(input.device)
            self.mask = self.mask.to(input.device)

        # Divide into sub-bands
        frames = frames.unsqueeze(2).repeat(1, 1, self.n_filters, 1)
        frames = frames * self.fbank[:, 0:-1]  # batch x num_frames x n_filters x frame_dim of 1.5 secs

        han_weight = torch.hann_window(self.cut, dtype=input.dtype, device=input.device)
        ham_weight = torch.hamming_window(self.cut, dtype=input.dtype, device=input.device)

        if self.num_chunks:

            # Divide computation to num_chunks along num_batch direction
            chunk_size = int(np.ceil(frames.shape[0] / self.num_chunks))
            frames = list(torch.split(frames, split_size_or_sections=chunk_size, dim=0))

            for chunk_idx in range(len(frames)):
                frames[chunk_idx], gain = self.compute_lpc(frames[chunk_idx],
                                                           self.order)  # batch x num_frames x n_filters x lpc_coeff
                if self.do_bwe:
                    frames[chunk_idx] = self.bwe_lpc_stabilizer(frames[chunk_idx])

                frames[chunk_idx] = self.compute_modspec_from_lpc(gain, frames[chunk_idx],
                                                                  self.coeff_num)  # batch x num_frames x n_filters x num_modspec

            frames = torch.cat(frames, dim=0)

        else:

            ### Compute all LPC in all bands and all frames parallely to make it fast

            frames, gain = self.compute_lpc(frames, self.order)  # batch x num_frames x n_filters x lpc_coeff
            if self.do_bwe:
                frames = self.bwe_lpc_stabilizer(frames)

            frames = self.compute_modspec_from_lpc(gain, frames,
                                                   self.coeff_num)  # batch x num_frames x n_filters x num_modspec
        modspec = frames

        modspec = modspec * self.mask  # (batch x num_frames x n_filters x num_modspec)

        modspec = [modspec * self.lifter[i].unsqueeze(0).unsqueeze(0).repeat(num_batch, num_frames, 1, 1) for i in
                   range(self.num_modulation_head)]  # (batch x num_frames x n_filters x num_modspec)
        modspec = torch.cat(modspec, axis=2)

        if self.complex_modulation:
            modspec = torch.fft.fft(modspec, 1 * int(round(
                self.fduration * self.frate)))  # (batch x num_frames x n_filters x int(self.fduration * self.frate))
        else:
            modspec = torch.fft.fft(modspec, 2 * int(round(
                self.fduration * self.frate)))  # (batch x num_frames x n_filters x int(self.fduration * self.frate))
        modspec = torch.abs(torch.exp(modspec))
        modspec = modspec[:, :, :, 0:self.cut] * han_weight / ham_weight
        modspec = torch.transpose(modspec, 2,
                                  3)  # (batch x num_frames x int(self.fduration * self.frate) x n_filters)

        # OVERLAP AND ADD
        modspec = self.OLA(modspec=modspec, t_samples=t_samples, dtype=input.dtype, device=input.device)

        if ilens is not None:
            olens = torch.floor(ilens * self.frate / self.srate)
            olens = olens.to(ilens.dtype)
            modspec.masked_fill_(make_pad_mask(olens, modspec, 1), 0.0000001)
            modspec = modspec[:, :torch.max(olens), :]
        else:
            olens = None

        return modspec, olens

    def forward(self, input: torch.Tensor, ilens: torch.Tensor = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute FDLP-Spectrogram forward function.

        Args:
            input: (Batch, Nsamples) or (Batch, Nsample, Channels)
            ilens: (Batch)
        Returns:
            output: (Batch, Frames, Freq) or (Batch, Frames, Channels, Freq)
        """

        bs = input.size(0)

        ft = False
        if self.stop_mmh_updates_after:
            ft = self.stop_mmh_updates_after <= self.num_updates  # Fine tune after ft is True
            if self.num_updates <= self.stop_mmh_updates_after:
                self.num_updates += 1

            if ft and self.mod_freq_ranges.requires_grad is True:
                logging.info('STOP UPDATING MMH....')
                self.mod_freq_ranges.requires_grad = False

        if input.dim() == 3:
            multi_channel = True
            # input: (Batch, Nsample, Channels) -> (Batch * Channels, Nsample)
            input = input.transpose(1, 2).reshape(-1, input.size(1))
        else:
            multi_channel = False
        # if ft:
        #    with torch.no_grad():
        output, olens = self.compute_spectrogram(input, ilens)
        # else:
        #    with contextlib.nullcontext():
        #        output, olens = self.compute_spectrogram(input, ilens)
        if multi_channel:
            # output: (Batch * Channel, Frames, Freq, 2=real_imag)
            # -> (Batch, Frame, Channel, Freq, 2=real_imag)
            output = output.view(bs, -1, output.size(1), output.size(2)).transpose(
                1, 2
            )

        return output, olens


class fdlp_spectrogram_dropout(fdlp_spectrogram):

    def __init__(self,
                 dropout_range_hz: str = '1,20',
                 dropout_width_hz: float = 1,
                 dropout_num: int = 3,
                 dropout_frame_num: int = 2,
                 return_nondropout_spectrogram: bool = False,
                 return_dropout_mask: bool = False,
                 dropout_while_eval: bool = False,
                 fixed_dropout: str = None,
                 pause_dropout_after_steps: int = None,
                 **kwargs
                 ):
        assert check_argument_types()
        super().__init__(**kwargs)
        dropout_range_hz = dropout_range_hz.strip().split(',')
        t_res = 1 / (self.fduration * 2)
        self.dropout_range_low = int(int(dropout_range_hz[0]) / t_res)
        self.dropout_range_high = int(int(dropout_range_hz[1]) / t_res)

        self.dropout_range = int(dropout_width_hz / t_res)
        self.dropout_num = dropout_num
        self.dropout_frame_num = dropout_frame_num
        self.return_nondropout_spectrogram = return_nondropout_spectrogram
        self.return_dropout_mask = return_dropout_mask
        if fixed_dropout:
            self.fixed_dropout = True
            fd = fixed_dropout.strip().split(',')
            self.fixed_dropout_lb = int(int(fd[0]) / t_res)
            self.fixed_dropout_ub = int(int(fd[1]) / t_res)
            self.fixed_dropout_lifter = self._generate_fixed_dropout_lifter()
        else:
            self.fixed_dropout = False
            self.fixed_dropout_lb = None
            self.fixed_dropout_ub = None
        self.dropout_while_eval = dropout_while_eval
        self.pause_dropout_after_steps = pause_dropout_after_steps
        if self.pause_dropout_after_steps:
            self.register_buffer("num_updates", torch.LongTensor([0]))
        else:
            self.num_updates = None

    def _generate_dropout_lifter(self, num_batch, num_frames, device):

        if self.fixed_dropout:
            lifter = self.fixed_dropout_lifter.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(num_batch, num_frames,
                                                                                             self.n_filters, 1)
            lifter = lifter.to(device)
        else:
            lifter = np.ones((num_batch, num_frames, self.coeff_num))
            for i in range(num_batch):
                for j in range(num_frames):
                    for idx in range(self.dropout_num):
                        r = random.randint(self.dropout_range_low, self.dropout_range_high)
                        lifter[i, j, r: r + self.dropout_range] = 0

            lifter = torch.tensor(lifter, dtype=self.datatype, device=device)
            lifter = lifter.unsqueeze(2).repeat(1, 1, self.n_filters, 1)
        return lifter

    def _generate_fixed_dropout_lifter(self, device='cpu'):
        lifter = np.ones(self.coeff_num)
        lifter[self.fixed_dropout_lb:self.fixed_dropout_ub] = 0
        lifter = torch.tensor(lifter, dtype=self.datatype, device=device)
        return lifter

    def compute_spectrogram(self, input: torch.Tensor, ilens: torch.Tensor = None) -> Tuple[
        torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Compute FDLP-Spectrogram With Matrices.

        Args:
            input: (Batch, Nsamples) or (Batch, Nsample)
            ilens: (Batch)
        Returns:
            output: (Batch, Frames, Freq) or (Batch, Frames, Freq)

        """
        if self.num_updates:
            self.num_updates += 1
            do_dropout = self.pause_dropout_after_steps >= self.num_updates
            if not do_dropout:
                logging.info("STOPPING DROPOUT FROM THIS ITER!")
        else:
            do_dropout = True

        t_samples = input.shape[1]
        num_batch = input.shape[0]

        # First divide the signal into frames
        t_samples, frames = self.get_frames(input)
        num_frames = frames.shape[1]

        # Get ids of frames to mask in each batch
        random_frame_idx = []
        batch_idx = np.arange(num_batch)
        if self.dropout_frame_num > num_frames:
            dpfn = num_frames
        else:
            dpfn = self.dropout_frame_num
        for idx in batch_idx:
            random_frame_idx.append(np.random.permutation(num_frames)[:dpfn])

        if self.complex_modulation:
            frames = torch.fft.ifft(frames) * int(self.srate * self.fduration)
        else:
            frames = self.dct_type2(frames) / np.sqrt(2 * int(self.srate * self.fduration))

        # Put fbank, mask and lifter into proper device if they are already not there
        if self.fbank.device.type != input.device.type:
            print('Transferring fbank, mask and lifter to {:s}'.format(input.device.type))
            self.fbank = self.fbank.to(input.device)
            self.lifter = self.lifter.to(input.device)
            self.mask = self.mask.to(input.device)

        # Compute features

        frames = frames.unsqueeze(2).repeat(1, 1, self.n_filters, 1)
        frames = frames * self.fbank[:, 0:-1]  # batch x num_frames x n_filters x frame_dim of 1.5 secs

        han_weight = torch.hann_window(self.cut, dtype=input.dtype, device=input.device)
        ham_weight = torch.hamming_window(self.cut, dtype=input.dtype, device=input.device)

        ### Compute all LPC in all bands and all frames parallely to make it fast

        frames, gain = self.compute_lpc(frames, self.order)  # batch x num_frames x n_filters x lpc_coeff

        if self.do_bwe:
            frames = self.bwe_lpc_stabilizer(frames)

        frames = self.compute_modspec_from_lpc(gain, frames,
                                               self.coeff_num)  # batch x num_frames x n_filters x num_modspec
        modspec = frames
        modspec = modspec * self.mask  # (batch x num_frames x n_filters x num_modspec)

        ## NOW COMPUTE 2 DIFFERENT FEATURES
        if self.return_nondropout_spectrogram:
            # Original modulation spectrum
            modspec_ori = modspec * self.lifter  # (batch x num_frames x n_filters x num_modspec)
            if self.complex_modulation:
                modspec_ori = torch.fft.fft(modspec_ori, 1 * int(
                    self.fduration * self.frate))  # (batch x num_frames x n_filters x int(self.fduration * self.frate))
            else:
                modspec_ori = torch.fft.fft(modspec_ori, 2 * int(
                    self.fduration * self.frate))  # (batch x num_frames x n_filters x int(self.fduration * self.frate))
            modspec_ori = torch.abs(torch.exp(modspec_ori))
            modspec_ori = modspec_ori[:, :, :, 0:self.cut] * han_weight / ham_weight
            modspec_ori = torch.transpose(modspec_ori, 2,
                                          3)  # (batch x num_frames x int(self.fduration * self.frate) x n_filters)

        if self.training or self.dropout_while_eval:
            # Do masking only during training
            if do_dropout:
                lifter_mask = self._generate_dropout_lifter(num_batch,
                                                            dpfn,
                                                            device=input.device)  # (batch x num_frames_reduced x n_filters x num_modspec)
                # Masked modulation spectrum
                for p, q in zip(batch_idx, random_frame_idx):
                    modspec[p, q, :, :] = modspec[p, q, :, :] * lifter_mask[p, :, :, :]

        if self.complex_modulation:
            modspec = torch.fft.fft(modspec, 1 * int(
                self.fduration * self.frate))  # (batch x num_frames x n_filters x int(self.fduration * self.frate))
        else:
            modspec = torch.fft.fft(modspec, 2 * int(
                self.fduration * self.frate))  # (batch x num_frames x n_filters x int(self.fduration * self.frate))
        modspec = torch.abs(torch.exp(modspec))
        modspec = modspec[:, :, :, 0:self.cut] * han_weight / ham_weight
        modspec = torch.transpose(modspec, 2, 3)  # (batch x num_frames x int(self.fduration * self.frate) x n_filters)

        if self.return_nondropout_spectrogram:
            feats_original = self.OLA(modspec=modspec_ori, t_samples=t_samples, dtype=input.dtype, device=input.device)
        feats = self.OLA(modspec=modspec, t_samples=t_samples, dtype=input.dtype, device=input.device)

        if ilens is not None:
            olens = torch.floor(ilens * self.frate / self.srate)
            olens = olens.to(ilens.dtype)
            if self.return_nondropout_spectrogram:
                feats_original.masked_fill_(make_pad_mask(olens, feats_original, 1), 0.0000001)
                feats_original = feats_original[:, :torch.max(olens), :]
            feats.masked_fill_(make_pad_mask(olens, feats, 1), 0.0000001)
            feats = feats[:, :torch.max(olens), :]
        else:
            olens = None

        if self.return_dropout_mask and self.return_nondropout_spectrogram:
            dropout_mask = torch.gt(torch.abs(feats_original - feats), 0)
        else:
            dropout_mask = None

        ## TODO: Fix this to return the same number of stuff
        if self.return_dropout_mask and self.return_nondropout_spectrogram:
            return feats, feats_original, olens, dropout_mask
        elif self.return_nondropout_spectrogram:
            return feats, feats_original, olens
        else:
            return feats, olens

    def forward(self, input: torch.Tensor, ilens: torch.Tensor = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute FDLP-Spectrogram forward function.

        Args:
            input: (Batch, Nsamples) or (Batch, Nsample, Channels)
            ilens: (Batch)
        Returns:
            output: (Batch, Frames, Freq) or (Batch, Frames, Channels, Freq)

        """
        bs = input.size(0)
        if input.dim() == 3:
            multi_channel = True
            # input: (Batch, Nsample, Channels) -> (Batch * Channels, Nsample)
            input = input.transpose(1, 2).reshape(-1, input.size(1))
        else:
            multi_channel = False
        if self.return_dropout_mask and self.return_nondropout_spectrogram:
            output, output_ori, olens, dropout_mask = self.compute_spectrogram(input, ilens)
        elif self.return_nondropout_spectrogram:
            output, output_ori, olens = self.compute_spectrogram(input, ilens)
        else:
            output, olens = self.compute_spectrogram(input, ilens)
        if multi_channel:
            # output: (Batch * Channel, Frames, Freq, 2=real_imag)
            # -> (Batch, Frame, Channel, Freq, 2=real_imag)
            output = output.view(bs, -1, output.size(1), output.size(2)).transpose(
                1, 2
            )
            if self.return_dropout_mask and self.return_nondropout_spectrogram:
                output_ori = output_ori.view(bs, -1, output_ori.size(1), output_ori.size(2)).transpose(
                    1, 2
                )
                dropout_mask = dropout_mask.view(bs, -1, dropout_mask.size(1), dropout_mask.size(2)).transpose(
                    1, 2
                )
            elif self.return_nondropout_spectrogram:
                output_ori = output_ori.view(bs, -1, output_ori.size(1), output_ori.size(2)).transpose(
                    1, 2
                )

        if self.return_dropout_mask and self.return_nondropout_spectrogram:
            return output, output_ori, olens, dropout_mask
        elif self.return_nondropout_spectrogram:
            return output, output_ori, olens
        else:
            return output, olens


class fdlp_spectrogram_update(fdlp_spectrogram):

    def levinson_durbin(self, R, p):
        """
        Levinson Durbin recursion to compute LPC coefficients

        :param R - autocorrelation coefficients: Tensor (batch x num_frames x n_filters x autocorr)
        :param p - lpc model order: int
        :return: Tensor (batch x num_frames x n_filters x lpc_coeff), Tensor (batch x num_frames x n_filters)

        """
        num_batch = R.shape[0]
        num_frames = R.shape[1]
        n_filters = R.shape[2]

        k = torch.zeros((num_batch, num_frames, n_filters, p), dtype=R.dtype, device=R.device)
        alphs = torch.zeros((num_batch, num_frames, n_filters, p, p), dtype=R.dtype, device=R.device)
        errs = torch.zeros((num_batch, num_frames, n_filters, p + 1), dtype=R.dtype, device=R.device)
        errs[:, :, :, 0] = R[:, :, :, 0]

        for i in range(1, p + 1):
            if i == 1:
                k[:, :, :, i - 1] = R[:, :, :, i].clone() / errs[:, :, :, i - 1].clone()
            else:
                k[:, :, :, i - 1] = (R[:, :, :, i] - torch.sum(
                    alphs[:, :, :, 0:i - 1, i - 2].clone() * torch.flip(R[:, :, :, 1:i].clone(), [3]), dim=3)) / errs[:,
                                                                                                                 :, :,
                                                                                                                 i - 1].clone()
            alphs[:, :, :, i - 1, i - 1] = k[:, :, :, i - 1]
            if i > 1:
                for j in range(1, i):
                    alphs[:, :, :, j - 1, i - 1] = alphs[:, :, :, j - 1, i - 2] - k[:, :, :, i - 1].clone() * alphs[:,
                                                                                                              :, :,
                                                                                                              i - j - 1,
                                                                                                              i - 2].clone()
            errs[:, :, :, i] = (1 - k[:, :, :, i - 1].clone() ** 2) * errs[:, :, :, i - 1].clone()

        return torch.cat((torch.ones((num_batch, num_frames, n_filters, 1), dtype=R.dtype, device=R.device),
                          -alphs[:, :, :, :, p - 1]), axis=3), errs[:, :, :, -1]

    def compute_lpc(self, input: torch.Tensor, order: int):

        """

        :param input: Tensor (batch x num_frames x n_filters x frame_dim)
        :return: Tensor (batch x num_frames x n_filters x lpc_coeff), Tensor (batch x num_frames x n_filters)
        """
        if self.precision_lpc:
            input = input.double()
        R = self.compute_autocorr(input)
        lpc_coeff, gain = self.levinson_durbin(R, p=order)
        if self.precision_lpc:
            lpc_coeff = lpc_coeff.float()
            gain = gain.float()

        return lpc_coeff, gain

    def compute_modspec_from_lpc(self, gain, lpc_coeff, lim):
        """
        :param gain: Tensor (batch x  num_frames x n_filters)
        :param lpc_coeff: Tensor (batch x num_frames x n_filters x lpc_num)
        :param lim: int
        :return: Tensor (batch x num_frames x n_filters x num_modspec),
        """

        num_batch = lpc_coeff.shape[0]
        num_frames = lpc_coeff.shape[1]
        n_filters = lpc_coeff.shape[2]
        lpc_coeff[:, :, :, 1:] = -lpc_coeff[:, :, :, 1:]
        lpc_cep = torch.zeros(num_batch, num_frames, n_filters, lim, dtype=lpc_coeff.dtype, device=lpc_coeff.device)

        lpc_cep[:, :, :, 0] = torch.log(torch.sqrt(gain))
        lpc_cep[:, :, :, 1] = lpc_coeff[:, :, :, 1]
        if lpc_coeff.shape[3] < lim:
            lpc_coeff = torch.cat(
                [lpc_coeff, torch.zeros(num_batch, num_frames, n_filters, int(lim - lpc_coeff.shape[3] + 1),
                                        device=lpc_coeff.device)], axis=3)
        for n in range(2, lim):
            a = torch.arange(1, n) / n
            a = a.to(lpc_coeff.device)
            b = torch.flip(lpc_coeff[:, :, :, 1:n], dims=[3])
            c = lpc_cep[:, :, :, 1:n]
            acc = torch.sum(a * b.clone() * c.clone(), axis=3)
            lpc_cep[:, :, :, n] = acc + lpc_coeff[:, :, :, n]
        return lpc_cep

    def get_frames(self, signal: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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

    def __warp_func_bark(self, x, warp_fact=1):
        import numpy as np
        return 6 * np.arcsinh((x / warp_fact) / 600)

    def initialize_filterbank(self, nfilters, nfft, srate, om_w=1, alp=1, fixed=1, bet=2.5, warp_fact=1):
        f_max = srate / 2
        warped_max = self.__warp_func_bark(f_max, warp_fact)
        fwarped_cf = np.linspace(0, warped_max, nfilters)
        f_linear = np.linspace(0, f_max, int(np.floor(nfft / 2 + 1)))
        f_warped = self.__warp_func_bark(f_linear, warp_fact)
        filts = np.zeros((nfilters, int(np.floor(nfft / 2 + 1))))
        alp_c = alp
        for i in range(nfilters):
            fc = fwarped_cf[i]
            if fixed == 1:
                alp = alp_c
            else:
                alp = alp_c * np.exp(-0.1 * fc)
            for j, fw in enumerate(f_warped):
                if fw - fc <= -om_w / 2:
                    filts[i, j] = np.power(10, alp * (fw - fc + om_w / 2))
                elif fw - fc > -om_w / 2 and fw - fc < om_w / 2:
                    filts[i, j] = 1
                else:
                    filts[i, j] = np.power(10, -bet * (fw - fc - om_w / 2))
        # return torch.tensor(filts, dtype=self.datatype)
        return torch.tensor(filts, dtype=self.datatype, device=self.device)

    def compute_spectrogram(self, input: torch.Tensor, ilens: torch.Tensor = None) -> Tuple[
        torch.Tensor, Optional[torch.Tensor]]:
        """Compute FDLP-Spectrogram With Matrices.

        Args:
            input: (Batch, Nsamples) or (Batch, Nsample)
            ilens: (Batch)
        Returns:
            output: (Batch, Frames, Freq) or (Batch, Frames, Freq)

        """

        t_samples = input.shape[1]
        num_batch = input.shape[0]

        # First divide the signal into frames
        frames = self.get_frames(input)
        num_frames = frames.shape[1]

        # Compute DCT (olens remains the same)
        frames = self.dct_type2(frames) / np.sqrt(2 * int(self.srate * self.fduration))

        # Put fbank, mask and lifter into proper device if they are already not there
        if self.mask.device.type != input.device.type:
            self.mask = self.mask.to(input.device)
            self.fbank = self.fbank.to(input.device)
            self.lifter = self.lifter.to(input.device)

        # Main loop to compute features

        ptr = int(0)
        frames = frames.unsqueeze(2).repeat(1, 1, self.n_filters, 1)
        frames = frames * self.fbank[:, 0:-1]  # batch x num_frames x n_filters x frame_dim of 1.5 secs

        han_weight = torch.hann_window(self.cut, dtype=input.dtype, device=input.device)
        ham_weight = torch.hamming_window(self.cut, dtype=input.dtype, device=input.device)

        ### Compute all LPC in all bands and all frames parallely to make it fast

        frames, gain = self.compute_lpc(frames, self.order)  # batch x num_frames x n_filters x lpc_coeff
        if self.do_bwe:
            frames = self.bwe_lpc_stabilizer(frames)

        frames = self.compute_modspec_from_lpc(gain, frames,
                                               self.coeff_num)  # batch x num_frames x n_filters x num_modspec
        modspec = frames
        modspec = modspec * self.mask  # (batch x num_frames x n_filters x num_modspec)
        modspec = modspec * self.lifter  # (batch x num_frames x n_filters x num_modspec)
        modspec = torch.fft.fft(modspec, 2 * int(
            self.fduration * self.frate))  # (batch x num_frames x n_filters x int(self.fduration * self.frate))
        modspec = torch.abs(torch.exp(modspec))
        modspec = modspec[:, :, :, 0:self.cut] * han_weight / ham_weight
        modspec = torch.transpose(modspec, 2, 3)  # (batch x num_frames x int(self.fduration * self.frate) x n_filters)

        ### Overlap and Add stage
        modspec = self.OLA(modspec=modspec, t_samples=t_samples, dtype=input.dtype, device=input.device)

        if ilens is not None:
            olens = torch.floor(ilens * self.frate / self.srate)
            olens = olens.to(ilens.dtype)
            modspec.masked_fill_(make_pad_mask(olens, modspec, 1), 0.0000001)
            modspec = modspec[:, :torch.max(olens), :]
        else:
            olens = None

        return modspec, olens


class fdlp_spectrogram_modnet(fdlp_spectrogram):

    def __init__(self,
                 dropout_frame_num: int = 2,
                 dropout_while_eval: bool = False,
                 pause_dropout_after_steps: int = None,
                 **kwargs
                 ):
        assert check_argument_types()
        super().__init__(**kwargs)

        self.dropout_frame_num = dropout_frame_num
        self.fixed_dropout_lb = None
        self.fixed_dropout_ub = None
        self.dropout_while_eval = dropout_while_eval
        self.pause_dropout_after_steps = pause_dropout_after_steps

    def _generate_zero_dropout_lifter(self, num_batch, num_frames, device):

        lifter = np.zeros(self.coeff_num)
        lifter = torch.tensor(lifter, dtype=self.datatype, device=device)
        lifter = lifter.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(num_batch, num_frames, self.n_filters, 1)
        lifter = lifter.to(device)

        return lifter

    def compute_spectrogram(self, input: torch.Tensor, ilens: torch.Tensor = None):
        """Compute FDLP-Spectrogram With Matrices.

        Args:
            input: (Batch, Nsamples) or (Batch, Nsample)
            ilens: (Batch)
        Returns:
            output: (Batch, Frames, Freq) or (Batch, Frames, Freq)

        """

        t_samples = input.shape[1]
        num_batch = input.shape[0]

        # First divide the signal into frames
        t_samples, frames = self.get_frames(input)
        num_frames = frames.shape[1]

        # Get ids of frames to mask in each batch. These are the features that will be dropped
        random_frame_idx = []
        batch_idx = np.arange(num_batch)

        if self.dropout_frame_num > num_frames:
            dpfn = num_frames
        else:
            dpfn = self.dropout_frame_num

        for idx in batch_idx:
            random_frame_idx.append(np.random.permutation(num_frames)[:dpfn])

        if self.complex_modulation:
            frames = torch.fft.ifft(frames) * int(self.srate * self.fduration)
        else:
            frames = self.dct_type2(frames) / np.sqrt(2 * int(self.srate * self.fduration))

        # Put fbank, mask and lifter into proper device if they are already not there
        if self.fbank.device.type != input.device.type:
            print('Transferring fbank, mask and lifter to {:s}'.format(input.device.type))
            self.fbank = self.fbank.to(input.device)
            self.lifter = self.lifter.to(input.device)
            self.mask = self.mask.to(input.device)

        # Compute features
        frames = frames.unsqueeze(2).repeat(1, 1, self.n_filters, 1)
        frames = frames * self.fbank[:, 0:-1]  # batch x num_frames x n_filters x frame_dim of 1.5 secs

        han_weight = torch.hann_window(self.cut, dtype=input.dtype, device=input.device)
        ham_weight = torch.hamming_window(self.cut, dtype=input.dtype, device=input.device)

        ### Compute all LPC in all bands and all frames parallely to make it fast

        frames, gain = self.compute_lpc(frames, self.order)  # batch x num_frames x n_filters x lpc_coeff

        if self.do_bwe:
            frames = self.bwe_lpc_stabilizer(frames)

        frames = self.compute_modspec_from_lpc(gain, frames,
                                               self.coeff_num)  # batch x num_frames x n_filters x num_modspec
        modspec = frames
        modspec = modspec * self.mask  # (batch x num_frames x n_filters x num_modspec)

        # Original modulation spectrum
        modspec_ori = modspec * self.lifter  # (batch x num_frames x n_filters x num_modspec)
        if self.complex_modulation:
            modspec_ori = torch.fft.fft(modspec_ori, 1 * int(
                self.fduration * self.frate))  # (batch x num_frames x n_filters x int(self.fduration * self.frate))
        else:
            modspec_ori = torch.fft.fft(modspec_ori, 2 * int(
                self.fduration * self.frate))  # (batch x num_frames x n_filters x int(self.fduration * self.frate))
        modspec_ori = torch.abs(torch.exp(modspec_ori))
        modspec_ori = modspec_ori[:, :, :, 0:self.cut] * han_weight / ham_weight
        modspec_ori = torch.transpose(modspec_ori, 2,
                                      3)  # (batch x num_frames x int(self.fduration * self.frate) x n_filters)

        if self.training or self.dropout_while_eval:
            # Do masking only during training
            lifter_mask = self._generate_zero_dropout_lifter(num_batch,
                                                             dpfn,
                                                             device=input.device)  # (batch x num_frames_reduced x n_filters x num_modspec)
            # Masked modulation spectrum
            for p, q in zip(batch_idx, random_frame_idx):
                modspec[p, q, :, :] = modspec[p, q, :, :] * lifter_mask[p, :, :, :]

        if self.complex_modulation:
            modspec = torch.fft.fft(modspec, 1 * int(
                self.fduration * self.frate))  # (batch x num_frames x n_filters x int(self.fduration * self.frate))
        else:
            modspec = torch.fft.fft(modspec, 2 * int(
                self.fduration * self.frate))  # (batch x num_frames x n_filters x int(self.fduration * self.frate))
        modspec = torch.abs(torch.exp(modspec))
        modspec = modspec[:, :, :, 0:self.cut] * han_weight / ham_weight
        modspec = torch.transpose(modspec, 2, 3)  # (batch x num_frames x int(self.fduration * self.frate) x n_filters)

        feats = self.OLA(modspec=modspec, t_samples=t_samples, dtype=input.dtype, device=input.device)

        if ilens is not None:
            olens = torch.floor(ilens * self.frate / self.srate)
            olens = olens.to(ilens.dtype)
            feats.masked_fill_(make_pad_mask(olens, feats, 1), 0.0000001)
            feats = feats[:, :torch.max(olens), :]
        else:
            olens = None

        return {'feats': feats, 'feats_original': modspec_ori, 'random_frame_idx': random_frame_idx}, olens

    def forward(self, input: torch.Tensor, ilens: torch.Tensor = None
                ):
        """Compute FDLP-Spectrogram forward function.

        Args:
            input: (Batch, Nsamples) or (Batch, Nsample, Channels)
            ilens: (Batch)
        Returns:
            output: (Batch, Frames, Freq) or (Batch, Frames, Channels, Freq)

        """
        bs = input.size(0)
        if input.dim() == 3:
            multi_channel = True
            # input: (Batch, Nsample, Channels) -> (Batch * Channels, Nsample)
            input = input.transpose(1, 2).reshape(-1, input.size(1))
        else:
            multi_channel = False

        output, olens = self.compute_spectrogram(input, ilens)

        return output, olens


class mvector(fdlp_spectrogram):

    def __init__(self,
                 lfr: float = 5,
                 **kwargs
                 ):
        assert check_argument_types()
        super().__init__(**kwargs)
        self.lfr = lfr

    def compute_spectrogram(self, input: torch.Tensor, ilens: torch.Tensor = None) -> Tuple[
        torch.Tensor, Optional[torch.Tensor]]:
        """Compute FDLP-Spectrogram With Matrices.

        Args:
            input: (Batch, Nsamples) or (Batch, Nsample)
            ilens: (Batch)
        Returns:
            output: (Batch, Frames, Freq) or (Batch, Frames, Freq)

        """
        num_batch = input.shape[0]
        if self.online_normalize:
            self.spectral_substraction_vector = self.get_normalizing_vector(input, fduration=25, overlap_fraction=0.98,
                                                                            append_len=500000, discont=np.pi)

        # First divide the signal into frames
        tsamples, frames = self.get_frames(input)
        num_frames = frames.shape[1]

        if self.spectral_substraction_vector is not None:
            self.spectral_substraction_vector = self.spectral_substraction_vector.to(input.device)
            # logging.info('Substracting spectral vector')
            frames = self.spectral_substraction_preprocessing(frames)

        if self.fbank.device.type != input.device.type:
            print('Transferring fbank to {:s}'.format(input.device.type))
            self.fbank = self.fbank.to(input.device)

        if self.num_chunks:
            chunk_size = int(np.ceil(frames.shape[1] / self.num_chunks))  # Number of frames
            frames = list(torch.split(frames, split_size_or_sections=chunk_size, dim=1))

            for idx in range(len(frames)):
                if self.complex_modulation:
                    frames[idx] = torch.fft.ifft(frames[idx]) * int(self.srate * self.fduration)
                else:
                    frames[idx] = self.dct_type2(frames[idx]) / np.sqrt(2 * int(self.srate * self.fduration))

                frames[idx] = frames[idx].unsqueeze(2).repeat(1, 1, self.n_filters, 1)
                frames[idx] = frames[idx] * self.fbank[:,
                                            0:-1]  # batch x num_frames x n_filters x frame_dim of 1.5 secs

                frames[idx], gain = self.compute_lpc(frames[idx],
                                                     self.order)  # batch x num_frames x n_filters x lpc_coeff
                if self.do_bwe:
                    frames[idx] = self.bwe_lpc_stabilizer(frames[idx])

                frames[idx] = self.compute_modspec_from_lpc(gain, frames[idx],
                                                            self.coeff_num)  # batch x num_frames x n_filters x num_modspec

                frames[idx] = frames[idx].reshape(frames[idx].size(0), frames[idx].size(1),
                                                  -1)  # batch x num_frames x n_filters * num_modspec
                if self.complex_modulation:
                    # frames[idx] = torch.log(torch.abs(frames[idx]))
                    frames[idx] = torch.cat(
                        [torch.view_as_real(frames[idx])[:, :, :, 0], torch.view_as_real(frames[idx])[:, :, :, 1]],
                        dim=-1)

            frames = torch.cat(frames, dim=1)
        else:
            # Compute DCT (olens remains the same)
            if self.complex_modulation:
                frames = torch.fft.ifft(frames) * int(self.srate * self.fduration)
            else:
                frames = self.dct_type2(frames) / np.sqrt(2 * int(self.srate * self.fduration))

            frames = frames.unsqueeze(2).repeat(1, 1, self.n_filters, 1)
            frames = frames * self.fbank[:, 0:-1]  # batch x num_frames x n_filters x frame_dim of 1.5 secs

            ### Compute all LPC in all bands and all frames parallely to make it fast

            frames, gain = self.compute_lpc(frames, self.order)  # batch x num_frames x n_filters x lpc_coeff
            if self.do_bwe:
                frames = self.bwe_lpc_stabilizer(frames)

            frames = self.compute_modspec_from_lpc(gain, frames,
                                                   self.coeff_num)  # batch x num_frames x n_filters x num_modspec

            frames = frames.reshape(frames.size(0), frames.size(1), -1)  # batch x num_frames x n_filters * num_modspec

            if self.complex_modulation:
                frames = torch.cat([torch.view_as_real(frames)[:, :, :, 0], torch.view_as_real(frames)[:, :, :, 1]],
                                   dim=-1)

        if self.lfr != self.frate:
            # We have to bilinear interpolate features to frame rate
            frames = frames.transpose(1, 2)
            frames = torch.nn.functional.interpolate(frames, scale_factor=np.ceil(self.frate / self.lfr), mode='bilinear')
            frames = frames.transpose(1, 2)

        if ilens is not None:
            olens = torch.floor(ilens * self.frate / self.srate)
            olens = olens.to(ilens.dtype)
            frames.masked_fill_(make_pad_mask(olens, frames, 1), 0.0000001)
            frames = frames[:, :torch.max(olens), :]
        else:
            olens = None

        return frames, olens

    def forward(self, input: torch.Tensor, ilens: torch.Tensor = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute FDLP-Spectrogram forward function.

        Args:
            input: (Batch, Nsamples) or (Batch, Nsample, Channels)
            ilens: (Batch)
        Returns:
            output: (Batch, Frames, Freq) or (Batch, Frames, Channels, Freq)

        """
        bs = input.size(0)
        if input.dim() == 3:
            multi_channel = True
            # input: (Batch, Nsample, Channels) -> (Batch * Channels, Nsample)
            input = input.transpose(1, 2).reshape(-1, input.size(1))
        else:
            multi_channel = False
        output, olens = self.compute_spectrogram(input, ilens)
        if multi_channel:
            # output: (Batch * Channel, Frames, Freq, 2=real_imag)
            # -> (Batch, Frame, Channel, Freq, 2=real_imag)
            output = output.view(bs, -1, output.size(1), output.size(2)).transpose(
                1, 2
            )

        return output, olens
