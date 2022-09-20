#!/usr/bin/env python3

import os
from espnet2.tasks.lm import LMTask


def get_parser():
    parser = LMTask.get_parser()


def main(cmd=None):
    """LM training.

    Example:

        % python lm_train.py asr --print_config --optim adadelta
        % python lm_train.py --config conf/train_asr.yaml
    """

    # Check if this runs in SageMaker based on existence of SageMaker-related file
    sagemaker_hp_json_path  = '/opt/ml/input/config/hyperparameters.json'
    if os.path.exists(sagemaker_hp_json_path):

        # Parse args for sagemaker; as it is hard to pass list directory,
        # args are passed as str and parsed as str
        import json
        with open(sagemaker_hp_json_path) as f:
            hp_json = json.load(f)
        cmd = eval(hp_json['cmd'])

        # SageMaker downloads data in /opt/ml/input/, but espnet assumes the data
        # in current directory. Create symblic link to /opt/ml/input/, which can be
        # known from evnrionment variables

        try:
            os.symlink(os.environ['SM_CHANNEL_DATA'], "data")
            os.symlink(os.environ['SM_CHANNEL_EXP'], "exp")
            os.symlink(os.environ['SM_CHANNEL_DUMP'], "dump")
            os.symlink(os.environ['SM_CHANNEL_CONF'], "conf")
        except FileExistsError:
            print("[Info] Symbolic link to data directory has been created before. This process continues by re-using the link.")
        except:
            raise

        LMTask.main(cmd=cmd)

    else:
        LMTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
