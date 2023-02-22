import copy
from typing import Optional
from typing import Tuple
from typing import Union
import logging

import humanfriendly
import numpy as np
import torch
from torch_complex.tensor import ComplexTensor
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.frontends.frontend import Frontend
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.layers.fdlp_spectrogram import fdlp_spectrogram, fdlp_spectrogram_update, fdlp_spectrogram_dropout, \
    fdlp_spectrogram_with_mmh, fdlp_spectrogram_modnet, mvector, fdlp_spectrogram_multiorder
from espnet2.utils.get_default_kwargs import get_default_kwargs


class RobustFrontend(AbsFrontend):
    """
    Robust front-end based on FDLP-spectrogram

    """

    def __init__(
            self,
            n_filters: int = 20,
            coeff_num: int = 80,
            coeff_range: str = '1,80',
            order: int = 80,
            fduration: float = 1.5,
            frate: int = 125,
            overlap_fraction: float = 0.50,
            srate: int = 16000,
            do_bwe: bool = False,
            bwe_factor: float = 0.8,
            bwe_iter_num: int = 1,
            precision_lpc: bool = True,
            update_fbank: bool = False,
            update_lifter: bool = False,
            update_lifter_multiband: bool = False,
            use_complex_lifter: bool = False,
            initialize_lifter: str = None,
            lifter_nonlinear_transformation: str = None,
            device: str = 'auto',
            modulation_dropout: bool = False,
            dropout_range_hz: str = '1,20',
            dropout_width_hz: float = 1,
            dropout_num: int = 3,
            dropout_frame_num: int = 1,
            return_nondropout_spectrogram: bool = False,
            return_dropout_mask: bool = False,
            fixed_dropout: str = '2,8',
            dropout_while_eval: bool = False,
            pause_dropout_after_steps: int = None,
            complex_modulation: bool = False,
            num_chunks: int = None,
            scale_lifter_gradient: float = None,
            boost_lifter_lr: float = 1,
            num_modulation_head: int = None,
            update_mmh: bool = True,
            stop_mmh_updates_after: int = None,
            log_scaled_modulation_filters: bool = True,
            freeze_lifter_finetune_updates: int = None,
            update_lifter_after_steps: int = None,
            fbank_config: str = '1,1,2.5',
            feature_batch: int = None,
            spectral_substraction_vector: str = None,
            dereverb_whole_sentence: bool = False,
            modnet: bool = False,
            online_normalize: bool = False,
            return_mvector: bool = False,
            lfr: float = 5,
            attach_mvector: bool = False,
            log_magnitude_modulation: bool = False,
            full_modulation_spectrum: bool = False,
            return_as_magnitude_phase: bool = False,
            multiorder: bool = False,
            order_list: str = '40,60,80,100',
            fs: Union[int, str] = 16000,
            frontend_conf: Optional[dict] = get_default_kwargs(Frontend),
    ):
        assert check_argument_types()
        super().__init__()

        # Deepcopy (In general, dict shouldn't be used as default arg)
        # frontend_conf = copy.deepcopy(frontend_conf)
        self.return_nondropout_spectrogram = return_nondropout_spectrogram
        self.return_dropout_mask = return_dropout_mask
        self.modulation_dropout = modulation_dropout
        self.num_modulation_head = num_modulation_head
        self.fduration = fduration
        self.frate = frate
        self.return_mvector = return_mvector
        self.coeff_num = coeff_num
        self.complex_modulation = complex_modulation

        if multiorder:
            self.fdlp_spectrogram = fdlp_spectrogram_multiorder(n_filters=n_filters, coeff_num=coeff_num,
                                                                coeff_range=coeff_range, order=order,
                                                                fduration=fduration, frate=frate,
                                                                overlap_fraction=overlap_fraction,
                                                                srate=srate, update_fbank=update_fbank,
                                                                use_complex_lifter=use_complex_lifter,
                                                                update_lifter=update_lifter,
                                                                update_lifter_multiband=update_lifter_multiband,
                                                                initialize_lifter=initialize_lifter,
                                                                complex_modulation=complex_modulation,
                                                                boost_lifter_lr=boost_lifter_lr,
                                                                num_chunks=num_chunks,
                                                                online_normalize=online_normalize,
                                                                scale_lifter_gradient=scale_lifter_gradient,
                                                                freeze_lifter_finetune_updates=freeze_lifter_finetune_updates,
                                                                update_lifter_after_steps=update_lifter_after_steps,
                                                                lifter_nonlinear_transformation=lifter_nonlinear_transformation,
                                                                fbank_config=fbank_config,
                                                                feature_batch=feature_batch,
                                                                spectral_substraction_vector=spectral_substraction_vector,
                                                                dereverb_whole_sentence=dereverb_whole_sentence,
                                                                do_bwe=do_bwe, bwe_factor=bwe_factor,
                                                                bwe_iter_num=bwe_iter_num,
                                                                order_list=order_list,
                                                                precision_lpc=precision_lpc, device=device)

        elif modnet:
            self.fdlp_spectrogram = fdlp_spectrogram_modnet(dropout_frame_num=dropout_frame_num,
                                                            dropout_while_eval=dropout_while_eval,
                                                            pause_dropout_after_steps=pause_dropout_after_steps,
                                                            fixed_dropout=fixed_dropout,
                                                            n_filters=n_filters, coeff_num=coeff_num,
                                                            coeff_range=coeff_range, order=order,
                                                            fduration=fduration, frate=frate,
                                                            overlap_fraction=overlap_fraction,
                                                            srate=srate, update_fbank=update_fbank,
                                                            update_lifter=update_lifter,
                                                            update_lifter_multiband=update_lifter_multiband,
                                                            lifter_nonlinear_transformation=lifter_nonlinear_transformation,
                                                            do_bwe=do_bwe, bwe_factor=bwe_factor,
                                                            bwe_iter_num=bwe_iter_num,
                                                            complex_modulation=complex_modulation,
                                                            precision_lpc=precision_lpc, device=device)
        elif num_modulation_head:
            self.fdlp_spectrogram = fdlp_spectrogram_with_mmh(n_filters=n_filters, coeff_num=coeff_num,
                                                              coeff_range=coeff_range, order=order,
                                                              fduration=fduration, frate=frate,
                                                              overlap_fraction=overlap_fraction,
                                                              srate=srate, update_fbank=update_fbank,
                                                              use_complex_lifter=use_complex_lifter,
                                                              complex_modulation=complex_modulation,
                                                              num_modulation_head=num_modulation_head,
                                                              update_mmh=update_mmh,
                                                              stop_mmh_updates_after=stop_mmh_updates_after,
                                                              log_scaled_modulation_filters=log_scaled_modulation_filters,
                                                              num_chunks=num_chunks,
                                                              fbank_config=fbank_config,
                                                              do_bwe=do_bwe, bwe_factor=bwe_factor,
                                                              bwe_iter_num=bwe_iter_num,
                                                              precision_lpc=precision_lpc, device=device)
        elif update_fbank:
            self.fdlp_spectrogram = fdlp_spectrogram_update(n_filters=n_filters, coeff_num=coeff_num,
                                                            coeff_range=coeff_range, order=order,
                                                            fduration=fduration, frate=frate,
                                                            overlap_fraction=overlap_fraction,
                                                            srate=srate, update_fbank=update_fbank,
                                                            update_lifter=update_lifter,
                                                            update_lifter_multiband=update_lifter_multiband,
                                                            use_complex_lifter=use_complex_lifter,
                                                            initialize_lifter=initialize_lifter,
                                                            lifter_nonlinear_transformation=lifter_nonlinear_transformation,
                                                            do_bwe=do_bwe, bwe_factor=bwe_factor,
                                                            bwe_iter_num=bwe_iter_num,
                                                            complex_modulation=complex_modulation,
                                                            precision_lpc=precision_lpc, device=device)
        elif modulation_dropout:
            self.fdlp_spectrogram = fdlp_spectrogram_dropout(dropout_range_hz=dropout_range_hz,
                                                             dropout_width_hz=dropout_width_hz,
                                                             dropout_num=dropout_num,
                                                             dropout_frame_num=dropout_frame_num,
                                                             return_nondropout_spectrogram=return_nondropout_spectrogram,
                                                             return_dropout_mask=return_dropout_mask,
                                                             fixed_dropout=fixed_dropout,
                                                             dropout_while_eval=dropout_while_eval,
                                                             pause_dropout_after_steps=pause_dropout_after_steps,
                                                             n_filters=n_filters, coeff_num=coeff_num,
                                                             coeff_range=coeff_range, order=order,
                                                             fduration=fduration, frate=frate,
                                                             overlap_fraction=overlap_fraction,
                                                             srate=srate, update_fbank=update_fbank,
                                                             update_lifter=update_lifter,
                                                             update_lifter_multiband=update_lifter_multiband,
                                                             lifter_nonlinear_transformation=lifter_nonlinear_transformation,
                                                             do_bwe=do_bwe, bwe_factor=bwe_factor,
                                                             bwe_iter_num=bwe_iter_num,
                                                             complex_modulation=complex_modulation,
                                                             feature_batch=feature_batch,
                                                             precision_lpc=precision_lpc, device=device)
        elif return_mvector:
            self.fdlp_spectrogram = mvector(n_filters=n_filters, coeff_num=coeff_num,
                                            coeff_range=coeff_range, order=order,
                                            fduration=fduration, frate=frate,
                                            overlap_fraction=overlap_fraction,
                                            srate=srate, update_fbank=update_fbank,
                                            use_complex_lifter=use_complex_lifter,
                                            update_lifter=update_lifter,
                                            update_lifter_multiband=update_lifter_multiband,
                                            initialize_lifter=initialize_lifter,
                                            complex_modulation=complex_modulation,
                                            boost_lifter_lr=boost_lifter_lr,
                                            num_chunks=num_chunks,
                                            online_normalize=online_normalize,
                                            scale_lifter_gradient=scale_lifter_gradient,
                                            freeze_lifter_finetune_updates=freeze_lifter_finetune_updates,
                                            lifter_nonlinear_transformation=lifter_nonlinear_transformation,
                                            fbank_config=fbank_config,
                                            feature_batch=feature_batch, lfr=lfr,
                                            log_magnitude_modulation=log_magnitude_modulation,
                                            full_modulation_spectrum=full_modulation_spectrum,
                                            spectral_substraction_vector=spectral_substraction_vector,
                                            dereverb_whole_sentence=dereverb_whole_sentence,
                                            return_as_magnitude_phase=return_as_magnitude_phase,
                                            do_bwe=do_bwe, bwe_factor=bwe_factor, bwe_iter_num=bwe_iter_num,
                                            precision_lpc=precision_lpc, device=device)
        else:
            self.fdlp_spectrogram = fdlp_spectrogram(n_filters=n_filters, coeff_num=coeff_num,
                                                     coeff_range=coeff_range, order=order,
                                                     fduration=fduration, frate=frate,
                                                     overlap_fraction=overlap_fraction,
                                                     srate=srate, update_fbank=update_fbank,
                                                     use_complex_lifter=use_complex_lifter,
                                                     update_lifter=update_lifter,
                                                     update_lifter_multiband=update_lifter_multiband,
                                                     initialize_lifter=initialize_lifter,
                                                     complex_modulation=complex_modulation,
                                                     boost_lifter_lr=boost_lifter_lr,
                                                     num_chunks=num_chunks,
                                                     online_normalize=online_normalize,
                                                     scale_lifter_gradient=scale_lifter_gradient,
                                                     freeze_lifter_finetune_updates=freeze_lifter_finetune_updates,
                                                     update_lifter_after_steps=update_lifter_after_steps,
                                                     lifter_nonlinear_transformation=lifter_nonlinear_transformation,
                                                     fbank_config=fbank_config,
                                                     feature_batch=feature_batch,
                                                     attach_mvector=attach_mvector,
                                                     spectral_substraction_vector=spectral_substraction_vector,
                                                     dereverb_whole_sentence=dereverb_whole_sentence,
                                                     do_bwe=do_bwe, bwe_factor=bwe_factor, bwe_iter_num=bwe_iter_num,
                                                     precision_lpc=precision_lpc, device=device)

        self.frontend = None
        self.n_filters = n_filters
        if update_lifter or update_lifter_multiband or num_modulation_head:
            self.pretrained_params = copy.deepcopy(self.fdlp_spectrogram.state_dict())
        else:
            self.pretrained_params = None

    def output_size(self) -> int:
        if self.return_mvector:
            return self.n_filters
        elif self.num_modulation_head:
            return self.n_filters * self.num_modulation_head
        else:
            return self.n_filters

    def reload_pretrained_parameters(self):
        if self.pretrained_params:
            self.fdlp_spectrogram.load_state_dict(self.pretrained_params)
            logging.info("Overwriting lifter parameters after initialization...")
        else:
            logging.info("No pretrained model provided...")

    def forward(
            self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # 1. Compute FDLP spectrogram
        if self.modulation_dropout:
            if self.return_dropout_mask and self.return_nondropout_spectrogram:
                input_spec, input_spec_ori, feats_lens, dropout_mask = self.fdlp_spectrogram(input, input_lengths)
            elif self.return_nondropout_spectrogram:
                input_spec, input_spec_ori, feats_lens = self.fdlp_spectrogram(input, input_lengths)
            else:
                input_spec, feats_lens = self.fdlp_spectrogram(input, input_lengths)
        else:

            input_spec, feats_lens = self.fdlp_spectrogram(input, input_lengths)

        # 3. [Multi channel case]: Select a channel
        # if input_spec.dim() == 4:
        #    # h: (B, T, C, F) -> h: (B, T, F)
        #    if self.training:
        #        # Select 1ch randomly
        #        ch = np.random.randint(input_spec.size(2))
        #        input_spec = input_spec[:, :, ch, :]
        #    else:
        #        # Use the first channel
        #        input_spec = input_spec[:, :, 0, :]

        if self.modulation_dropout:
            if self.return_dropout_mask and self.return_nondropout_spectrogram:
                return input_spec, input_spec_ori, feats_lens, dropout_mask
            elif self.return_nondropout_spectrogram:
                return input_spec, input_spec_ori, feats_lens
            else:
                return input_spec, feats_lens
        else:
            return input_spec, feats_lens
