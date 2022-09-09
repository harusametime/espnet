#!/usr/bin/env python3
from espnet2.tasks.tts import TTSTask
import os

def get_parser():
    parser = TTSTask.get_parser()
    return parser


def main(cmd=None):
    """TTS training

    Example:

        % python tts_train.py asr --print_config --optim adadelta
        % python tts_train.py --config conf/train_asr.yaml
    """

    if os.path.exists('/opt/ml/input/config/hyperparameters.json'):
        import argparse
        parser = argparse.ArgumentParser(description='Process some integers.')
        parser.add_argument('--cmd', type=str, default="")
        args = parser.parse_args()

        TTSTask.main(cmd=args.cmd)
    else:
        TTSTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
