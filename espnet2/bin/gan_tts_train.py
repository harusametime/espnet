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
            os.symlink(os.environ['SM_CHANNEL_DOWNLOADS'], "downloads")
        except FileExistsError:
            print("[Info] Symbolic link to data directory has been created before. This process continues by re-using the link.")
        except:
            raise

        GANTTSTask.main(cmd=cmd)
    else:
        GANTTSTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
