import argparse
import torch
from argparse import Namespace

from datasets import CIFAR10
from datasets import MNIST
from datasets import SHANGHAITECH
from datasets import UCSDPed2,UCSDPed1,UCSDPed2_deepSVDD
from models import LSACIFAR10
from models import LSAMNIST
from models import LSAShanghaiTech
from models import LSAUCSD, LSAUCSD_deepSVDD
from result_helpers import OneClassResultHelper
from result_helpers import VideoAnomalyDetectionResultHelper, \
    VideoAnomalyDetectionResultHelper_deepSVDD
from utils import set_random_seed
#
# 我自己开发的
from datasets import UCSDPed2_deepSVDD, UCSDPed2_deepSVDD_TRAIN
from models import LSAUCSD_deepSVDD
from result_helpers import VideoAnomalyDetectionResultHelper_deepSVDD
import os, utils

def test_mnist():
    # type: () -> None
    """
    Performs One-class classification tests on MNIST
    """

    # Build dataset and model
    dataset = MNIST(path='data/MNIST')
    model = LSAMNIST(input_shape=dataset.shape, code_length=64, cpd_channels=100).to(device).eval()

    # Set up result helper and perform test
    helper = OneClassResultHelper(dataset, model, checkpoints_dir='checkpoints/mnist/', output_file='mnist.txt')
    helper.test_one_class_classification()


def test_cifar():
    # type: () -> None
    """
    Performs One-class classification tests on CIFAR
    """

    # Build dataset and model
    dataset = CIFAR10(path='data/CIFAR10')
    model = LSACIFAR10(input_shape=dataset.shape, code_length=64, cpd_channels=100).to(device).eval()

    # Set up result helper and perform test
    helper = OneClassResultHelper(dataset, model, checkpoints_dir='checkpoints/cifar10/', output_file='cifar10.txt')
    helper.test_one_class_classification()


def test_ucsdped1():
    # type: () -> None
    """
    Performs video anomaly detection tests on UCSD Ped2.
    """

    # Build dataset and model
    dataset = UCSDPed1(path=Config.video_folder)
    model = LSAUCSD(input_shape=dataset.shape,
                    code_length=64, cpd_channels=100).to(device).eval()

    # Set up result helper and perform test
    helper = VideoAnomalyDetectionResultHelper(dataset, model,
                                               checkpoint=Config.model_ckpt,
                                               output_file=Config.output_file)
    helper.test_video_anomaly_detection()


def test_ucsdped2():
    # type: () -> None
    """
    Performs video anomaly detection tests on UCSD Ped2.
    """

    dataset_name = "ucsd_ped2"
    #
    # Load the "reference" parameters from parent_dir json file
    # args = parser.parse_args()
    parent_dir = "test_results"
    json_path = "checkpoints/ped2_test.json"
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)
    # Launch job (name has to be unique)
    performance_testing(parent_dir, dataset_name, params)


def test_shanghaitech():
    # type: () -> None
    """
    Performs video anomaly detection tests on ShanghaiTech.
    """

    # Build dataset and model
    dataset = SHANGHAITECH(path=Config.video_folder)
    model = LSAShanghaiTech(input_shape=dataset.shape,
                            code_length=64, cpd_channels=100).to(device).eval()

    # Set up result helper and perform test
    helper = VideoAnomalyDetectionResultHelper(dataset,
                                               model,
                                               checkpoint=Config.model_ckpt,
                                               output_file=Config.output_file)
    helper.test_video_anomaly_detection()

######################################################################

#
def performance_testing(parent_dir, job_name, params):
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

    # # Write parameters in json file
    # json_path = os.path.join(model_dir, 'params.json')
    # params.save(json_path)

    # Launch training with this config
    dataset_train = UCSDPed2_deepSVDD_TRAIN(path=params.test_dir)
    dataset_eval = UCSDPed2_deepSVDD(path=params.test_dir)
    model = LSAUCSD_deepSVDD(input_shape=dataset_train.shape,
                             code_length=params.code_length)
    # Set up result helper and perform test
    helper = VideoAnomalyDetectionResultHelper_deepSVDD(
                    dataset_train, dataset_eval, model, params, model_dir)
    # mutli_task train，这次使用逻辑上的分阶段训练：代码实现上用权重来控制
    # 先给rec loss极高的权重，只训练rec loss；然后只训练 probability loss
    # 最后恢复权重，联合训练 rec loss + deep_SVDD loss
    helper.test_video_anomaly_detection() # 执行 train


def test_vis_graph():
    from tensorboardX import SummaryWriter
    # 可视化网络结构
    dataset = UCSDPed1(path=Config.video_folder) # 所以 Config也要写对
    model = LSAUCSD(input_shape=dataset.shape,
                    code_length=64, cpd_channels=100).to(device).eval()
    model_input = torch.rand([1380, 1, 8, 32, 32]).to(device)
    with SummaryWriter(log_dir="./summary/graphs/LSAUCSD") as writer:
        writer.add_graph(model, input_to_model=model_input)

def test_get_c_init():
    import os,time
    from datasets import UCSDPed2_TRAIN

    dataset = UCSDPed2_TRAIN(path="data/UCSD_Anomaly_Dataset.v1p2")  # 需重写
    net = LSAUCSD(input_shape=dataset.shape,
                    code_length=64, cpd_channels=100).to(device).eval()
    # 偷个懒，暂时直接使用作者预训练模型，后面再改成 AE预训练
    checkpoint = "checkpoints/ucsd_ped2.pkl"
    if os.path.exists(checkpoint):
        print("{} load !".format(checkpoint))
        net.load_w(checkpoint)
    # 只保留 Encoder部分
    # net = "" # 不需要，因为直接输出 z (latent vector) 就 OK 了
    #
    batch_size = 2
    c = init_center_c(dataset, net, batch_size)
    # Use torch.save(tensor, 'file.pt') and torch.load('file.pt')
    torch.save(c, "c_init_ped2.pt")
    cl = torch.load("c_init_ped2.pt")
    print("c1.shape: ", cl.shape)

# init c
def init_center_c(dataset, net, batch_size, eps=0.1):
    """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    n_samples = 0
    c = torch.zeros(128, device=device) # for ped2, 暂时是求均值，且是 2*64=128

    # print("(dataset.train_videos: ", (dataset.train_videos))
    for cl_idx, video_id in enumerate(dataset.train_videos):
        # print("cl_idex: ", cl_idx)
        # Run the test
        dataset.train(video_id)
        loader = DataLoader(dataset,
                            collate_fn=dataset.collate_fn,
                            num_workers=4,
                            batch_size=2, # 最大能支持的batch_size
                            shuffle=False)
        #
        with torch.no_grad():
            for i, (x, y) in tqdm(enumerate(loader),desc="get_c_init of ped2"):
                #
                x = x.to(device)

                x_r, z, z_dist = net(x) # z就是我需要的 latent vector, (batchsize, seq_len, out_features)

                # print("z.shape: ", z.shape, z.shape[0]) # (1380,2,64)
                # 事实是：每个clip被处理为 690 个patch,
                # 每个32x32x1的patch的embedding 是(2,64) 的tensor，
                # 因为这里batch_size=2,所以就是2个clip，即 2*690 = 1380 个 patch
                z = z.view(-1, 690, 128) # ?个clip，每个clip有690个patch，每个patch的embedding为128-dim vector
                z = torch.mean(z, dim=1)
                print("z.size: ", z.size()) # [2, 88320] # for ped2, 16 frames: 88320 = 690 * 2 * 64
                n_samples += z.shape[0]
                c += torch.sum(z, dim=0)

    c /= n_samples

    # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps

    return c


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

    # Run test
    if args.dataset == 'mnist':
        test_mnist()
    elif args.dataset == 'cifar10':
        test_cifar()
    elif args.dataset == 'ucsd-ped2':
        test_ucsdped2()
    elif args.dataset == 'shanghaitech':
        test_shanghaitech()
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')


# Entry point
if __name__ == '__main__':
    # 做测试简单点
    # test_mnist()
    # test_cifar()
    # test_ucsdped1()
    test_ucsdped2()
    # test_shanghaitech()
    # test_vis_graph()
    #
    # test_get_c_init()
    # main()