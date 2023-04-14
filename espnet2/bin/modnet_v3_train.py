#!/usr/bin/env python3
from espnet2.tasks.modnet_v3 import ModnetTask_v3


def get_parser():
    parser = ModnetTask_v3.get_parser()
    return parser


def main(cmd=None):
    r"""ASR training.

    Example:s

        % python asr_train.py asr --print_config --optim adadelta \
                > conf/train_asr.yaml
        % python asr_train.py --config conf/train_asr.yaml
    """
    ModnetTask_v3.main(cmd=cmd)


if __name__ == "__main__":
    main()