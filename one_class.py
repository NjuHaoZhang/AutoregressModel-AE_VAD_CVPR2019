from os.path import join
from typing import Tuple

import numpy as np
import torch
from prettytable import PrettyTable
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.base import OneClassDataset
from models.base import BaseModule
from models.loss_functions import LSALoss
from utils import novelty_score
from utils import normalize
#
from torch import optim # 优化器 by HaoZhang
from tensorboardX import SummaryWriter
import os,time
import matplotlib.pyplot as plt
plt.switch_backend('agg')

#----------------------------------------------------------------#
# 引入配置信息
# from config import Config_mnist_training as Config
from config import Config_cifar10_training as Config
#
# from config import Config_mnist_testing as Config
#from config import Config_cifar10_testing as Config
#
device_idx = Config.device_idx
device = torch.device("cuda:" + device_idx) # 配置使用的GPU
#----------------------------------------------------------------#

class OneClassResultHelper(object):
    """
    Performs tests for one-class datasets (MNIST or CIFAR-10).
    """

    def __init__(self, dataset, model, checkpoints_dir, output_file):
        # type: (OneClassDataset, BaseModule, str, str) -> None
        """
        Class constructor.

        :param dataset: dataset class.
        :param model: pytorch model to evaluate.
        :param checkpoints_dir: directory holding checkpoints for the model.
        :param output_file: text file where to save results.
        """
        self.dataset = dataset
        self.model = model
        self.checkpoints_dir = checkpoints_dir
        self.output_file = output_file

        # Set up loss function
        self.loss = LSALoss(cpd_channels=100)

    # for Training
    def train_one_class_classification(self):
        # type: () -> None
        """
        Actually performs tests.
        """

        # Load the checkpoint
        if os.path.exists(self.checkpoints_dir):
            print("{} load !".format(self.checkpoints_dir))
            self.model.load_w(self.checkpoints_dir)

        # optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=Config.LR)
        optimizer.zero_grad()

        # Start iteration over classes
        train_cnt_step = []
        for cl_idx, cl in enumerate(self.dataset.test_classes):
            train_cnt_step.append(0) # 为cl维持各自的计数器
        with SummaryWriter(log_dir="summary/train_{0}_lr={1}_lam={2}_{3}".format(
                Config.dataset_name, Config.LR,
                Config.LAM, Config.normal_or_dist),
                comment="{}".format(Config.dataset_name)) as writer:
            for epoch in range(Config.epoch):
                for cl_idx, cl in enumerate(self.dataset.test_classes):

                    # Run the actual test
                    self.dataset.train(cl)
                    loader = DataLoader(self.dataset,
                                        num_workers=Config.num_workers,
                                        batch_size=Config.batch_size,
                                        shuffle=Config.shuffle)

                    for i, (x, y) in tqdm(enumerate(loader), desc=f'Training for {self.dataset}'):
                        train_cnt_step[cl] = train_cnt_step[cl] + 1
                        #
                        x = x.to(device)
                        x_r, z, z_dist = self.model(x)
                        total_loss_bp = self.loss(x, x_r, z, z_dist)  # 返回的是一个tensor
                        # print("z, z_dist: ", z, z_dist)
                        reconstruction_loss = self.loss.reconstruction_loss
                        autoregression_loss = self.loss.autoregression_loss
                        total_loss = self.loss.total_loss
                        #
                        print("total_loss_{}: {}".format(cl, total_loss))  #
                        if train_cnt_step[cl] % Config.plot_every == 0:
                            writer.add_scalars("train_loss_{}".format(cl),
                                               {'total_loss': total_loss,
                                                'reconstruction_loss':
                                                    reconstruction_loss,
                                                'autoregression_loss':
                                                    autoregression_loss
                                                },
                                               train_cnt_step[cl])
                        if train_cnt_step[cl] % Config.save_ckpt_every == 0:
                            # 保存模型 （在每个epoch结束时保存）# 或者根据 cnt_step设置
                            ckpt_path = '{prefix}{dataset}_{anoclass}_{time}.pkl'.format(
                                prefix=Config.prefix,
                                dataset=Config.dataset_name,
                                anoclass=cl,
                                time=time.strftime('%m%d_%H%M')  # 这个要和下面 save()无限近
                            )
                            torch.save(self.model.state_dict(), ckpt_path)
                        #
                        total_loss_bp.backward() # BP
                        # 梯度裁剪
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                       max_norm=20)
                        optimizer.step()
                print("epoch {} complete !".format(epoch))
                # 做完一个epoch training，开始一个 val ??
                # 因为代码结构是：每次更新c1就要重新set dataset，所以重置dataset做val是OK的
                # val_cnt_step = self.evaluate(cl, writer, val_cnt_step) # TODO 写的有问题

    def evaluate(self, cl, writer, val_cnt_step):
        # type: (int) -> Tuple[float, float, float, float]
        """
        Computes normalizing coeffients for the computation of the Novelty score (Eq. 9-10).
        :param cl: the class to be considered normal.
        :return: a tuple of normalizing coefficients in the form (llk_min, llk_max, rec_min, rec_max).
        """
        self.dataset.val(cl)
        loader = DataLoader(self.dataset,
                            num_workers=Config.num_workers,
                            batch_size=Config.batch_size,
                            shuffle=Config.shuffle)

        for i, (x, y) in enumerate(loader):
            val_cnt_step = val_cnt_step + 1
            #
            x = x.to(device)
            x_r, z, z_dist = self.model(x)
            self.loss(x, x_r, z, z_dist)
            autoregression_loss =  self.loss.autoregression_loss
            reconstruction_loss =  self.loss.reconstruction_loss
            total_loss = self.loss.total_loss
            # if val_cnt_step % Config.plot_every == 0: # 好像步数也不多
            writer.add_scalars("val_loss",
                               {'total_loss': total_loss,
                                'reconstruction_loss': reconstruction_loss,
                                'autoregression_loss': autoregression_loss
                                },
                               val_cnt_step)

        return val_cnt_step


    ###########################################################################

    @torch.no_grad()
    def test_one_class_classification(self):
        # type: () -> None
        """
        Actually performs tests.
        """

        # Prepare a table to show results
        oc_table = self.empty_table

        # Set up container for metrics from all classes
        all_metrics = []

        # Start iteration over classes
        for cl_idx, cl in enumerate(self.dataset.test_classes):

            # Load the checkpoint
            self.model.load_w(join(self.checkpoints_dir, f'{cl}.pkl'))

            # First we need a run on validation, to compute
            # normalizing coefficient of the Novelty Score (Eq.9)
            min_llk, max_llk, min_rec, max_rec = self.compute_normalizing_coefficients(cl)

            # Run the actual test
            self.dataset.test(cl)
            loader = DataLoader(self.dataset)

            sample_llk = np.zeros(shape=(len(loader),))
            sample_rec = np.zeros(shape=(len(loader),))
            sample_y = np.zeros(shape=(len(loader),))
            for i, (x, y) in tqdm(enumerate(loader), desc=f'Computing scores for {self.dataset}'):
                x = x.to(device)

                x_r, z, z_dist = self.model(x)

                self.loss(x, x_r, z, z_dist)

                sample_llk[i] = - self.loss.autoregression_loss
                sample_rec[i] = - self.loss.reconstruction_loss
                sample_y[i] = y.item()

            # Normalize scores
            sample_llk = normalize(sample_llk, min_llk, max_llk)
            sample_rec = normalize(sample_rec, min_rec, max_rec)

            # Compute the normalized novelty score
            sample_ns = novelty_score(sample_llk, sample_rec)

            # Compute AUROC for this class
            this_class_metrics = [
                roc_auc_score(sample_y, sample_llk),  # likelihood metric
                roc_auc_score(sample_y, sample_rec),  # reconstruction metric
                roc_auc_score(sample_y, sample_ns)    # novelty score
            ]
            oc_table.add_row([cl_idx] + this_class_metrics)

            all_metrics.append(this_class_metrics)

        # Compute average AUROC and print table
        all_metrics = np.array(all_metrics)
        avg_metrics = np.mean(all_metrics, axis=0)
        oc_table.add_row(['avg'] + list(avg_metrics))
        print(oc_table)

        # Save table
        with open(self.output_file, mode='w') as f:
            f.write(str(oc_table))


    def compute_normalizing_coefficients(self, cl):
        # type: (int) -> Tuple[float, float, float, float]
        """
        Computes normalizing coeffients for the computation of the Novelty score (Eq. 9-10).
        :param cl: the class to be considered normal.
        :return: a tuple of normalizing coefficients in the form (llk_min, llk_max, rec_min, rec_max).
        """
        self.dataset.val(cl)
        loader = DataLoader(self.dataset)

        sample_llk = np.zeros(shape=(len(loader),))
        sample_rec = np.zeros(shape=(len(loader),))
        for i, (x, y) in enumerate(loader):
            x = x.to(device)

            x_r, z, z_dist = self.model(x)

            self.loss(x, x_r, z, z_dist)

            sample_llk[i] = - self.loss.autoregression_loss
            sample_rec[i] = - self.loss.reconstruction_loss

        return sample_llk.min(), sample_llk.max(), sample_rec.min(), sample_rec.max()

    @property
    def empty_table(self):
        # type: () -> PrettyTable
        """
        Sets up a nice ascii-art table to hold results.
        This table is suitable for the one-class setting.
        :return: table to be filled with auroc metrics.
        """
        table = PrettyTable()
        table.field_names = ['Class', 'AUROC-LLK', 'AUROC-REC', 'AUROC-NS']
        table.float_format = '0.3'
        return table
