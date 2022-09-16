#!/usr/bin/env python3
from espnet2.tasks.modnet_v2 import ModnetTask_v2


def get_parser():
    parser = ModnetTask_v2.get_parser()
    return parser


def main(cmd=None):
    r"""ASR training.

    Example:s

        % python asr_train.py asr --print_config --optim adadelta \
                > conf/train_asr.yaml
        % python asr_train.py --config conf/train_asr.yaml
    """
    ModnetTask_v2.main(cmd=cmd)


if __name__ == "__main__":
    main()