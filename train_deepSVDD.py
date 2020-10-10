import argparse
import torch
from argparse import Namespace

from datasets import CIFAR10,CIFAR10_TRAIN
from datasets import MNIST,MNIST_TRAIN
from datasets import SHANGHAITECH,SHANGHAITECH_TRAIN
from datasets import UCSDPed2_TRAIN
from models import LSACIFAR10
from models import LSAMNIST
from models import LSAShanghaiTech
from models import LSAUCSD
from result_helpers import OneClassResultHelper
from result_helpers import VideoAnomalyDetectionResultHelper
from utils import set_random_seed
#
# 我自己开发的
from datasets import UCSDPed2_deepSVDD, UCSDPed2_deepSVDD_TRAIN, UCSDPed1_deepSVDD, UCSDPed1_deepSVDD_TRAIN
from models import LSAUCSD_deepSVDD
from result_helpers import VideoAnomalyDetectionResultHelper_deepSVDD
#
import utils
import os

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

    json_path = "params_logs/ped1/params_AE.json" # for AE
    # json_path = "params_logs/ped1/params_calculate_c.json"  # for calculate_cvim tr

    params = utils.Params(json_path)
    # print("params.model_ckpt: ", params.model_ckpt)
    # print("params.restore_file: ", params.restore_file)
    model_save_dir = "checkpoints/ped1"  # 模型未来保存的位置
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    # Launch training with this config
    extra_paras_dict = {'train_window_size': params.train_window_size,
                        'test_window_size': params.test_window_size}
    dataset_train = UCSDPed1_deepSVDD_TRAIN(path=params.train_dir,
                                            exrta_paras_dict=extra_paras_dict)
    dataset_eval = UCSDPed1_deepSVDD(path=params.test_dir,
                                     exrta_paras_dict=extra_paras_dict)
    model = LSAUCSD_deepSVDD(input_shape=dataset_train.shape,
                             code_length=params.code_length)
    # Set up result helper and perform test
    helper = VideoAnomalyDetectionResultHelper_deepSVDD(
        dataset_train, dataset_eval, model, params)
    # mutli_task train，这次使用逻辑上的分阶段训练：代码实现上用权重来控制
    # 先给rec loss极高的权重，只训练rec loss；然后只训练 probability loss
    # 最后恢复权重，联合训练 rec loss + deep_SVDD loss
    helper.train_video_anomaly_detection()  # 需重写

def train_UCSDped2():
    # type: () -> None
    """
    Performs video anomaly detection tests on UCSD Ped2.
    """
    # json_path = "params_logs/ped2/params_AE.json" # for AE
    json_path = "params_logs/ped2/params_calculate_c.json" # for calculate_cvim tr

    params = utils.Params(json_path)
    # print("params.model_ckpt: ", params.model_ckpt)
    # print("params.restore_file: ", params.restore_file)
    model_save_dir = "checkpoints/ped2" # 模型未来保存的位置
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    # Launch training with this config
    extra_paras_dict = {'train_window_size': params.train_window_size,
                        'test_window_size': params.test_window_size}
    dataset_train = UCSDPed2_deepSVDD_TRAIN(path=params.train_dir,
                                            exrta_paras_dict=extra_paras_dict)
    dataset_eval = UCSDPed2_deepSVDD(path=params.test_dir,
                                     exrta_paras_dict=extra_paras_dict)
    model = LSAUCSD_deepSVDD(input_shape=dataset_train.shape,
                             code_length=params.code_length)
    # Set up result helper and perform test
    helper = VideoAnomalyDetectionResultHelper_deepSVDD(
        dataset_train, dataset_eval, model, params)
    # mutli_task train，这次使用逻辑上的分阶段训练：代码实现上用权重来控制
    # 先给rec loss极高的权重，只训练rec loss；然后只训练 probability loss
    # 最后恢复权重，联合训练 rec loss + deep_SVDD loss
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
    train_UCSDped1()
    # train_UCSDped2()
    # train_shanghaitech()
    #
    # python train.py ucsd-ped2
    # nohup python train.py >UCSDped2.out &
    #
    # main()
    # nohup python train.py ucsd-ped2 >ucsd-ped2.out &
    # nohup python train.py shanghaitech >shanghaitech.out &