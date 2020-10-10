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
#
# 我自己开发的
from datasets import UCSDPed2_deepSVDD, UCSDPed2_deepSVDD_TRAIN, UCSDPed1_deepSVDD, UCSDPed1_deepSVDD_TRAIN
from models import LSAUCSD_deepSVDD
from result_helpers import VideoAnomalyDetectionResultHelper_deepSVDD
#
import argparse
from subprocess import check_call
import sys, os
import utils
#
import argparse
import json
import os
from tabulate import tabulate


# PYTHON = sys.executable
# parser = argparse.ArgumentParser()
# parser.add_argument('--parent_dir', default='experiments/learning_rate',
#                     help='Directory containing params.json')
# parser.add_argument('--data_dir', default='data/64x64_SIGNS',
#                     help="Directory containing the dataset")
# device = torch.device("cuda:" + device_idx)  # 配置使用的GPU


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
    dataset_name = "ucsd_ped2"
    #
    # lam_svdd_opt(dataset_name)
    # window_size_opt(dataset_name)
    latent_code_size_opt(dataset_name)

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

def lam_svdd_opt(dataset_name):
    # Load the "reference" parameters from parent_dir json file
    # args = parser.parse_args()
    parent_dir = os.path.join("parameters_optimization/lam_svdd",
                              dataset_name)
    json_path = os.path.join(parent_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Perform hypersearch over one parameter
    lam_svdd_list = [1, 10, 50, 100] # 自定义搜索范围，这个策略可优化，TODO

    for lam_svdd in lam_svdd_list:
        # Modify the relevant parameter in params
        params.lam_svdd = lam_svdd

        # Launch job (name has to be unique)
        job_name = "lam_svdd_{}".format(lam_svdd)
        launch_training_job(parent_dir, job_name, params)
    #
    # 汇总结果
    synthesize_results(parent_dir)

def window_size_opt(dataset_name):
    # Load the "reference" parameters from parent_dir json file
    # args = parser.parse_args()
    parent_dir = os.path.join("parameters_optimization/window_size",
                              dataset_name)
    json_path = os.path.join(parent_dir, 'params.json')
    assert os.path.isfile(json_path), \
        "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Perform hypersearch over one parameter
    window_size_list = [12, 20, 24] # 自定义搜索范围，这个策略可优化，TODO

    for window_size in window_size_list:
        # Modify the relevant parameter in params
        params.train_window_size = window_size
        params.test_window_size = window_size

        # Launch job (name has to be unique)
        job_name = "window_size_{}".format(window_size)
        launch_training_job(parent_dir, job_name, params)
    #
    # 汇总结果
    synthesize_results(parent_dir)

def latent_code_size_opt(dataset_name):
    # Load the "reference" parameters from parent_dir json file
    # args = parser.parse_args()
    parent_dir = os.path.join("parameters_optimization/latent_code_len",
                              dataset_name)
    json_path = os.path.join(parent_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Perform hypersearch over one parameter
    # code_length_list = [32, 256] # 3号卡跑. 自定义搜索范围，这个策略可优化，TODO,
    code_length_list = [128] # 0号卡跑

    params.device = "0" #手动设置当前的device_id

    for latent_code_len in code_length_list:
        # Modify the relevant parameter in params
        params.code_length = latent_code_len
        params.restore_file = "checkpoints/{}/code_length_{}.pth.tar".format(dataset_name, str(latent_code_len))
        #
        # Launch job (name has to be unique)
        job_name = "latent_code_len_{}".format(latent_code_len)
        launch_training_job(dataset_name, parent_dir, job_name, params)
    #
    # 汇总结果
    synthesize_results(parent_dir)

def launch_training_job(dataset_name, parent_dir, job_name, params):
    """Launch training of the model with a set of hyperparameters in parent_dir/job_name
    Args:
        model_dir: (string) directory containing config, weights and log
        data_dir: (string) directory containing the dataset
        params: (dict) containing hyperparameters
    """
    # Create a new folder in parent_dir with unique_name "job_name"
    model_dir = os.path.join(parent_dir, job_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Write parameters in json file
    json_path = os.path.join(model_dir, 'params.json')
    params.model_dir = model_dir
    params.save(json_path)

    # Launch training with this config
    exrta_paras_dict = {'train_window_size':params.train_window_size,
                        'test_window_size': params.test_window_size}
    if dataset_name == "ucsd_ped1":
        dataset_train = UCSDPed1_deepSVDD_TRAIN(path=params.train_dir,
            exrta_paras_dict=exrta_paras_dict)
        dataset_eval = UCSDPed1_deepSVDD(path=params.test_dir,
            exrta_paras_dict=exrta_paras_dict)
    if dataset_name == "ucsd_ped2":
        dataset_train = UCSDPed2_deepSVDD_TRAIN(path=params.train_dir,
                                                exrta_paras_dict=exrta_paras_dict)
        dataset_eval = UCSDPed2_deepSVDD(path=params.test_dir,
                                         exrta_paras_dict=exrta_paras_dict)
    # mutli_task train，这次使用逻辑上的分阶段训练：代码实现上用权重来控制
    # 先给rec loss极高的权重，只训练rec loss；然后只训练 probability loss
    # 最后恢复权重，联合训练 rec loss + deep_SVDD loss
    model = LSAUCSD_deepSVDD(input_shape=dataset_train.shape,
                             code_length=params.code_length)
    # Set up result helper and perform test
    helper = VideoAnomalyDetectionResultHelper_deepSVDD(
        dataset_train, dataset_eval, model, params)
    helper.hyperparas_search()


def synthesize_results(parent_dir):
    """Aggregates results from the metrics_eval_best_weights.json in a parent folder"""
    # Aggregate metrics from args.parent_dir directory
    metrics = dict()
    aggregate_metrics(parent_dir, metrics) # metrics在函数里面被修改
    #
    table = metrics_to_table(metrics)
    # Display the table to terminal
    print(table)

    # Save results in parent_dir/results.md
    save_file = os.path.join(parent_dir, "results.md")
    with open(save_file, 'w') as f:
        f.write(table)

def aggregate_metrics(parent_dir, metrics):
    """Aggregate the metrics of all experiments in folder `parent_dir`.
    Assumes that `parent_dir` contains multiple experiments, with their results stored in
    `parent_dir/subdir/metrics_dev.json`
    Args:
        parent_dir: (string) path to directory containing experiments results
        metrics: (dict) subdir -> {'accuracy': ..., ...}
    """
    # Get the metrics for the folder if it has results from an experiment
    metrics_file = os.path.join(parent_dir, 'metrics_val_best_weights.json')
    if os.path.isfile(metrics_file):
        with open(metrics_file, 'r') as f:
            metrics[parent_dir] = json.load(f)

    # Check every subdirectory of parent_dir
    for subdir in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, subdir)):
            continue
        else:
            aggregate_metrics(os.path.join(parent_dir, subdir), metrics)

def metrics_to_table(metrics):
    # Get the headers from the first subdir. Assumes everything has the same metrics
    headers = metrics[list(metrics.keys())[0]].keys()
    table = [[subdir] + [values[h] for h in headers] for subdir, values in metrics.items()]
    res = tabulate(table, headers, tablefmt='pipe')

    return res

################################################################################################
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
##############################################

# Entry point
if __name__ == '__main__':
    train_UCSDped2()