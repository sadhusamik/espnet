from typing import Tuple

import numpy as np
import torch
from typeguard import check_argument_types

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.asr.frontend.s3prl import S3prlFrontend
from espnet2.asr.frontend.robust import RobustFrontend


class FusedFrontends(AbsFrontend):
    def __init__(
            self, frontends=None, align_method="linear_projection", interpolate=False, dpout=0, proj_dim=100, fs=16000
    ):
        assert check_argument_types()
        super().__init__()
        self.align_method = (
            align_method  # fusing method : linear_projection only for now
        )
        self.proj_dim = proj_dim  # dim of the projection done on each frontend
        self.frontends = []  # list of the frontends to combine
        self.interpolate = interpolate
        self.dpout = dpout
        for i, frontend in enumerate(frontends):
            frontend_type = frontend["frontend_type"]
            if frontend_type == "default":
                n_mels, fs, n_fft, win_length, hop_length = (
                    frontend.get("n_mels", 80),
                    fs,
                    frontend.get("n_fft", 512),
                    frontend.get("win_length"),
                    frontend.get("hop_length", 128),
                )
                window, center, normalized, onesided = (
                    frontend.get("window", "hann"),
                    frontend.get("center", True),
                    frontend.get("normalized", False),
                    frontend.get("onesided", True),
                )
                fmin, fmax, htk, apply_stft = (
                    frontend.get("fmin", None),
                    frontend.get("fmax", None),
                    frontend.get("htk", False),
                    frontend.get("apply_stft", True),
                )

                self.frontends.append(
                    DefaultFrontend(
                        n_mels=n_mels,
                        n_fft=n_fft,
                        fs=fs,
                        win_length=win_length,
                        hop_length=hop_length,
                        window=window,
                        center=center,
                        normalized=normalized,
                        onesided=onesided,
                        fmin=fmin,
                        fmax=fmax,
                        htk=htk,
                        apply_stft=apply_stft,
                    )
                )
            elif frontend_type == "robust":
                n_filters, srate, coeff_num, coeff_range, order, fduration, frate, overlap_fraction, return_mvector, lfr, num_chunks, complex_modulation = (
                    frontend.get("n_filters", 20),
                    frontend.get("srate", 16000),
                    frontend.get("coeff_num", 80),
                    frontend.get("coeff_range", '0,80'),
                    frontend.get("order", 80),
                    frontend.get("fduration", 1.5),
                    frontend.get("frate", 100),
                    frontend.get("overlap_fraction", 0.5),
                    frontend.get("return_mvector", False),
                    frontend.get("lfr", 5),
                    frontend.get("num_chunks", 2),
                    frontend.get("complex_modulation", False),
                )

                self.frontends.append(
                    RobustFrontend(
                        n_filters=n_filters,
                        coeff_num=coeff_num,
                        coeff_range=coeff_range,
                        srate=srate,
                        order=order,
                        fduration=fduration,
                        frate=frate,
                        num_chunks=num_chunks,
                        overlap_fraction=overlap_fraction,
                        complex_modulation=complex_modulation,
                        return_mvector=return_mvector,
                        lfr=lfr,
                    )
                )
            elif frontend_type == "s3prl":
                frontend_conf, download_dir, multilayer_feature = (
                    frontend.get("frontend_conf"),
                    frontend.get("download_dir"),
                    frontend.get("multilayer_feature"),
                )
                self.frontends.append(
                    S3prlFrontend(
                        fs=fs,
                        frontend_conf=frontend_conf,
                        download_dir=download_dir,
                        multilayer_feature=multilayer_feature,
                    )
                )

            else:
                raise NotImplementedError  # frontends are only default or s3prl

        self.frontends = torch.nn.ModuleList(self.frontends)
        if self.interpolate:
            self.gcd = None
            self.factors = [1, 1]
        else:
            self.gcd = np.gcd.reduce([frontend.hop_length for frontend in self.frontends])
            self.factors = [frontend.hop_length // self.gcd for frontend in self.frontends]

        if torch.cuda.is_available():
            dev = "cuda"
        else:
            dev = "cpu"
        if self.align_method == "linear_projection":
            if self.interpolate:
                self.projection_layers = [
                    torch.nn.Linear(
                        in_features=frontend.output_size(),
                        out_features=self.proj_dim,
                    )
                    for i, frontend in enumerate(self.frontends)
                ]
            else:
                self.projection_layers = [
                    torch.nn.Linear(
                        in_features=frontend.output_size(),
                        out_features=self.factors[i] * self.proj_dim,
                    )
                    for i, frontend in enumerate(self.frontends)
                ]
            self.projection_layers = torch.nn.ModuleList(self.projection_layers)
            self.projection_layers = self.projection_layers.to(torch.device(dev))

    def output_size(self) -> int:
        return len(self.frontends) * self.proj_dim

    def forward(
            self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # step 0 : get all frontends features
        feats = []
        for frontend in self.frontends:
            with torch.no_grad():
                input_feats, feats_lens = frontend.forward(input, input_lengths)
            feats.append([input_feats, feats_lens])

        save_shape = feats[0][1]
        if (
                self.align_method == "linear_projection"
        ):  # TODO(Dan): to add other align methods
            # first step : projections
            for i, frontend in enumerate(self.frontends):
                feats[i] = self.projection_layers[i](feats[i][0])

            # 2nd step : reshape or interpolate
            if self.interpolate:
                shape_list = [x.shape[1] for x in feats]
                m = max(shape_list)
                m_idx = shape_list.index(m)
                for i, F in enumerate(feats):
                    if i != m_idx:
                        feats[i] = feats[i].transpose(1, 2)
                        mlen = feats[i].shape[2]
                        scl = m / mlen
                        feats[i] = torch.nn.functional.interpolate(feats[i], scale_factor=scl, mode='linear')
                        feats[i] = feats[i].transpose(1, 2)
            else:
                for i, frontend in enumerate(self.frontends):
                    bs, nf, dim = feats[i].shape
                    feats[i] = torch.reshape(
                        feats[i], (bs, nf * self.factors[i], dim // self.factors[i])
                    )

            # 3rd step : drop the few last frames
            m = min([x.shape[1] for x in feats])
            feats = [x[:, :m, :] for x in feats]

            if self.dpout != 0:
                num_batch = feats[0].shape[0]
                for i in range(num_batch):
                    if np.random.rand() > 1 - self.dpout:
                        stream = np.random.randint(0, 2)
                        feats[stream][i, :, :] *= 0

            feats = torch.cat(
                feats, dim=-1
            )  # change the input size of the preencoder : proj_dim * n_frontends
            feats_lens = torch.ones_like(save_shape) * (m)
        else:
            raise NotImplementedError

        return feats, feats_lens
