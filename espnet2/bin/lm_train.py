#!/usr/bin/env python3

import os
from espnet2.tasks.lm import LMTask


def get_parser():
    parser = LMTask.get_parser()
    return parser


def main(cmd=None):
    """LM training.

    Example:

        % python lm_train.py asr --print_config --optim adadelta
        % python lm_train.py --config conf/train_asr.yaml
    """

    if os.path.exists('/opt/ml/input/config/hyperparameters.json'):
        import argparse
        parser = argparse.ArgumentParser(description='Process some integers.')
        parser.add_argument('--cmd', type=str, default="")
        args = parser.parse_args()

        LMTask.main(cmd=args.cmd)
    else:
        LMTask.main(cmd=cmd)

if __name__ == "__main__":
    main()
