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

        # Parse args for sagemaker; as it is hard to pass list directory,
        # args are passed as str and parsed as str
        import argparse
        parser = argparse.ArgumentParser(description='Process some integers.')
        parser.add_argument('--cmd', type=str, default="")
        args = parser.parse_args()
        cmd = eval(args.cmd)
        cmd.append("sagemaker")

        # SageMaker downloads data in /opt/ml/input/, but espnet assumes the data
        # in current directory. Create symblic link to /opt/ml/input/, which can be
        # known from evnrionment variables
        print(os.listdir('.'))
        os.symlink(os.environ['SM_CHANNEL_DATA'], ".")
        os.symlink(os.environ['SM_CHANNEL_EXP'], ".")
        os.symlink(os.environ['SM_CHANNEL_DUMP'], ".")
        os.symlink(os.environ['SM_CHANNEL_CONF'], ".")

        LMTask.main(cmd=cmd)
    else:
        LMTask.main(cmd=cmd)

if __name__ == "__main__":
    main()
