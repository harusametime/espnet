#!/usr/bin/env python3
from espnet2.tasks.asr import ASRTask


def get_parser():
    parser = ASRTask.get_parser()
    return parser


def main(cmd=None):
    r"""ASR training.

    Example:

        % python asr_train.py asr --print_config --optim adadelta \
                > conf/train_asr.yaml
        % python asr_train.py --config conf/train_asr.yaml
    """
    if os.path.exists('/opt/ml/input/config/hyperparameters.json'):

        # Parse args for sagemaker; as it is hard to pass list directory,
        # args are passed as str and parsed as str
        import argparse
        parser = argparse.ArgumentParser(description='SageMaker and Espnet args')
        parser.add_argument('--cmd', type=str, default="")
        args = parser.parse_args()
        cmd = eval(args.cmd)
        cmd.append("sagemaker")

        # SageMaker downloads data in /opt/ml/input/, but espnet assumes the data
        # in current directory. Create symblic link to /opt/ml/input/, which can be
        # known from evnrionment variables
        os.symlink(os.environ['SM_CHANNEL_DATA'], "data")
        os.symlink(os.environ['SM_CHANNEL_EXP'], "exp")
        os.symlink(os.environ['SM_CHANNEL_DUMP'], "dump")
        os.symlink(os.environ['SM_CHANNEL_CONF'], "conf")

        ASRTask.main(cmd=cmd)

    else:
        ASRTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
