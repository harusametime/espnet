#!/usr/bin/env python3
from espnet2.tasks.gan_tts import GANTTSTask
import os


def get_parser():
    parser = GANTTSTask.get_parser()
    return parser


def main(cmd=None):
    """GAN-based TTS training

    Example:

        % python gan_tts_train.py --print_config --optim1 adadelta
        % python gan_tts_train.py --config conf/train.yaml
    """

    if os.path.exists('/opt/ml/input/config/hyperparameters.json'):
        import argparse
        parser = argparse.ArgumentParser(description='Process some integers.')
        parser.add_argument('--cmd', type=str, default="")
        args = parser.parse_args()
        GANTTSTask.main(cmd=cmd)
    else:
        GANTTSTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
