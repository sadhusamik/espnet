# ESPNet Automatic Speech Recognition(ASR) with FDLP spectrogram 

This is a fork of the ESPNet toolkit with a FDLP-spectrograms ([**Radically Old Way of Computing Spectra: Applications in End-to-End ASR**](https://arxiv.org/abs/2103.14129)) computed on-the-fly front-end that can have trainable components 


* The main implementation of FDLP spectrgrams can be found [here](https://github.com/sadhusamik/espnet/blob/master/espnet2/layers/fdlp_spectrogram.py)
* This is part of the [robust](https://github.com/sadhusamik/espnet/blob/master/espnet2/asr/frontend/robust.py) ASR front-end.

The important parameters accepted by [robust](https://github.com/sadhusamik/espnet/blob/master/espnet2/asr/frontend/robust.py) front-end are 

```python
n_filters: int = 20,       # Number of filters
coeff_num: int = 80,       # Number of modulaiton coefficients to compute at resolution of 1/fduration 
coeff_range: str = '1,80', # Which coefficients to keep for spectrogram 
order: int = 80,           # FDLP model order 
fduration: float = 1.5,    # Window length
frate: int = 125,          # Frame rate
overlap_fraction: float = 0.5,  # Overlap fraction of windows to do Overlap-add
complex_modulation: bool = True, # Use complex FDLP 
fbank_config: str = '1,1,2.5', # width 1 bark , low-frequency slope 1, high frequency slope 2.5
```

### Complex FDLP 

* Paper: [Complex Frequency Domain Linear Prediction: A Tool to Compute Modulation Spectrum of Speech](https://arxiv.org/abs/2203.13216)

* All that you would need to do is set ```complex_modulation=True```  in the robust ASR front-end.

* A sample config file for ASR training with REVERB data can be found [here](https://github.com/sadhusamik/espnet/blob/master/egs2/reverb/asr1/conf/tuning/train_asr_transformer4_robustfrontend.yaml) that can be run with [this script](https://github.com/sadhusamik/espnet/blob/master/egs2/reverb/asr1/run_frontend.sh). In this config we exclude specAugment and speed perturbation. 

### Importance of Different Temporal Modulations of Speech 

* Paper: [Importance of Different Temporal Modulations of Speech: A Tale of Two Perspectives](https://arxiv.org/abs/2204.00065)
* To update the modulation weights (lifter weights) you would need to set ```update_lifter_multiband=True``` and ```lifter_nonlinear_transformation='relu'``` in the "robust" ASR front-end.
* A sample config file for ASR training with updatable modulation weights and REVERB data can be found [here](https://github.com/sadhusamik/espnet/blob/master/egs2/reverb/asr1/conf/tuning/train_asr_transformer4_robustfrontend_updated.yaml) that can be run with [this script](https://github.com/sadhusamik/espnet/blob/master/egs2/reverb/asr1/run_frontend_updated.sh). In this config we exclude specAugment and speed perturbation.