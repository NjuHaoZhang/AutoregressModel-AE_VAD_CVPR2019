import argparse
import torch
from argparse import Namespace

from datasets import CIFAR10,CIFAR10_TRAIN
from datasets import MNIST,MNIST_TRAIN
from datasets import SHANGHAITECH,SHANGHAITECH_TRAIN
from datasets import UCSDPed2_TRAIN,UCSDPed1_TRAIN
from models import LSACIFAR10
from models import LSAMNIST
from models import LSAShanghaiTech
from models import LSAUCSD
from result_helpers import OneClassResultHelper
from result_helpers import VideoAnomalyDetectionResultHelper
from utils import set_random_seed

#--------------------------------------------------------------#
# 引入配置信息
# from config import Config_mnist_training as Config
# from config import Config_cifar10_training as Config
#
from config import Config_ped1_training as Config
# from config import Config_ped2_training as Config
#
# from config import Config_shanghaitech_training as Config
#
device_idx = Config.device_idx
device = torch.device("cuda:" + device_idx)  # 配置使用的GPU
#--------------------------------------------------------------#


def train_mnist():
    # type: () -> None
    """
    Performs One-class classification tests on MNIST
    """

    # Build dataset and model
    dataset = MNIST_TRAIN(path=Config.video_folder)
    model = LSAMNIST(input_shape=dataset.shape, code_length=64,
                     cpd_channels=100).to(device).train()

    # Set up result helper and perform test
    helper = OneClassResultHelper(dataset, model,
                                  checkpoints_dir=Config.model_ckpt,
                                  output_file='mnist.txt')
    helper.train_one_class_classification()


def train_cifar10():
    # type: () -> None
    """
    Performs One-class classification tests on CIFAR
    """

    # Build dataset and model
    dataset = CIFAR10_TRAIN(path=Config.video_folder)
    model = LSACIFAR10(input_shape=dataset.shape, code_length=64,
                       cpd_channels=100).to(device).train()

    # Set up result helper and perform test
    helper = OneClassResultHelper(dataset, model,
                                  checkpoints_dir=Config.model_ckpt,
                                  output_file='cifar10.txt')
    helper.train_one_class_classification()

def train_UCSDped1():
    # type: () -> None
    """
    Performs video anomaly detection tests on UCSD Ped2.
    """

    # Build dataset and model
    dataset = UCSDPed1_TRAIN(path=Config.video_folder) # 需重写
    model = LSAUCSD(input_shape=dataset.shape,
                    code_length=64, cpd_channels=100).train() # 注意 train()

    # Set up result helper and perform test
    helper = VideoAnomalyDetectionResultHelper(dataset, model,
                                               checkpoint=Config.model_ckpt,
                                               output_file=Config.output_file)
    helper.train_video_anomaly_detection() # 需重写

def train_UCSDped2():
    # type: () -> None
    """
    Performs video anomaly detection tests on UCSD Ped2.
    """

    # Build dataset and model
    dataset = UCSDPed2_TRAIN(path=Config.video_folder) # 需重写
    model = LSAUCSD(input_shape=dataset.shape,
                    code_length=64, cpd_channels=100).train() # 注意 train()

    # Set up result helper and perform test
    helper = VideoAnomalyDetectionResultHelper(dataset, model,
                                               checkpoint=Config.model_ckpt,
                                               output_file=Config.output_file)
    helper.train_video_anomaly_detection() # 需重写


def train_shanghaitech():

    # type: () -> None
    """
    Performs video anomaly detection tests on ShanghaiTech.
    """

    # Build dataset and model
    # 加入分布式
    dataset = SHANGHAITECH_TRAIN(path=Config.video_folder)
    model = LSAShanghaiTech(input_shape=dataset.shape,
                            code_length=64, cpd_channels=100).train()
    # 下面的model处理写到train逻辑里面去

    # Set up result helper and perform test
    helper = VideoAnomalyDetectionResultHelper(dataset,
                                               model,
                                               checkpoint=Config.model_ckpt,
                                               output_file=Config.output_file)
    helper.train_video_anomaly_detection()


def parse_arguments():
    # type: () -> Namespace
    """
    Argument parser.

    :return: the command line arguments.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset', type=str,
                        help='The name of the dataset to perform tests on.'
                             'Choose among `mnist`, `cifar10`, `ucsd-ped2`, `shanghaitech`', metavar='')

    return parser.parse_args()


def main():

    # Parse command line arguments
    args = parse_arguments()

    # Lock seeds
    set_random_seed(30101990)

    # # Run test
    # if args.dataset == 'mnist':
    #     test_mnist()
    # elif args.dataset == 'cifar10':
    #     test_cifar()
    if args.dataset == 'ucsd-ped2':
        train_UCSDped2()

    elif args.dataset == 'shanghaitech':
        train_shanghaitech()
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')


# Entry point
if __name__ == '__main__':
    # # 暂时简单点直接运行看效果，再来优化代码
    # train_mnist()
    # train_cifar10()
    # train_UCSDped1()
    train_UCSDped2()
    # train_shanghaitech()
    #
    # python train.py ucsd-ped2
    # nohup python train.py >UCSDped2.out &
    #
    # main()
    # nohup python train.py ucsd-ped2 >ucsd-ped2.out &
    # nohup python train.py shanghaitech >shanghaitech.out &
