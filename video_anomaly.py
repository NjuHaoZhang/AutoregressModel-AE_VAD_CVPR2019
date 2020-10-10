from typing import Tuple

import numpy as np
import torch
from prettytable import PrettyTable
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.base import VideoAnomalyDetectionDataset
from models.base import BaseModule
from models.loss_functions import LSALoss
from utils import normalize
from utils import novelty_score
#
from torch import optim # 优化器 by HaoZhang
from tensorboardX import SummaryWriter
import os,time
import matplotlib.pyplot as plt
plt.switch_backend('agg')

#-------------------------------------------------------------#
# 引入配置信息
# from config import Config_ped1_training as Config
from config import Config_ped2_training as Config
# from config import Config_shanghaitech_training as Config
#
# from config import Config_ped1_testing as Config
# from config import Config_ped2_testing as Config
# from config import Config_ShanghaiTech_testing as Config
#
device_idx = Config.device_idx
device = torch.device("cuda:" + device_idx) # 配置使用的GPU
#-------------------------------------------------------------#


class ResultsAccumulator:
    """
    Accumulates results in a buffer for a sliding window
    results computation. Employed to get frame-level scores
    from clip-level scores.
    ` In order to recover the anomaly score of each
    frame, we compute the mean score of all clips in which it
    appears`
    """
    def __init__(self, time_steps):
        # type: (int) -> None
        """
        Class constructor.

        :param time_steps: the number of frames each clip holds.
        """

        # This buffers rotate.
        self._buffer = np.zeros(shape=(time_steps,), dtype=np.float32)
        self._counts = np.zeros(shape=(time_steps,))

    def push(self, score):
        # type: (float) -> None
        """
        Pushes the score of a clip into the buffer.
        :param score: the score of a clip

        # push()的；逻辑是：
        每次新来一个clip的loss，就让 _buffer的16个元素都加上这个clip的loss
        并且自增 num_clip
        """

        # Update buffer and counts
        # print("self._buffer_before: ", self._buffer)
        self._buffer += score # 一个向量加上常数，等价于：[0,0,0 ...] + 1 = [1,1,1...]
        # print("self._buffer_after: ", self._buffer)
        self._counts += 1 # 同_buffer

    def get_next(self):
        # type: () -> float
        """
        Gets the next frame (the first in the buffer) score,
        computed as the mean of the clips in which it appeared,
        and rolls the buffers.

        :return: the averaged score of the frame exiting the buffer.

        # get_next()的逻辑是：
        首先取_buffer[0]/_counts[0]作为当前clip的第一帧的loss
            ,因为_buffer[0]累加了_counts[0]个clip的loss
        #
        然后_buffer左移1，实现的效果有：_buffer[0]移至[-1]然后被置零，其他元素左移1格
        这样做的物理意义：
        我在纸上已经做了模拟运算，但是why it work ? i do not have idea.
        """

        # Return first in buffer
        ret = self._buffer[0] / self._counts[0]

        # Roll time backwards
        self._buffer = np.roll(self._buffer, shift=-1)
        self._counts = np.roll(self._counts, shift=-1)

        # Zero out final frame (next to be filled)
        self._buffer[-1] = 0
        self._counts[-1] = 0

        return ret

    @property
    def results_left(self):
        # type: () -> np.int32
        """
        Returns the number of frames still in the buffer.
        """
        return np.sum(self._counts != 0).astype(np.int32)


class VideoAnomalyDetectionResultHelper(object):
    """
    Performs tests for video anomaly detection datasets (UCSD Ped2 or Shanghaitech).
    """

    def __init__(self, dataset, model, checkpoint, output_file):
        # type: (VideoAnomalyDetectionDataset, BaseModule, str, str) -> None
        """
        Class constructor.

        :param dataset: dataset class.
        :param model: pytorch model to evaluate.
        :param checkpoint: path of the checkpoint for the model.
        :param output_file: text file where to save results.
        """
        self.dataset = dataset
        self.model = model
        self.checkpoint = checkpoint
        self.output_file = output_file

        # Set up loss function
        self.loss = LSALoss(cpd_channels=Config.cpd_channels, lam=Config.LAM)

    ####################################################################################
    # by HaoZhang
    # TODO 如果 Datastet_train 没有，就模仿写一个。当然，首选问作者要code
    def train_video_anomaly_detection(self):
        # type: () -> None
        """
        训练
        """

        c, t, h, w = self.dataset.raw_shape

        # Load the checkpoint
        if os.path.exists(self.checkpoint):
            print("{} load !".format(self.checkpoint))
            self.model.load_w(self.checkpoint)
        # 下面是分布式版本
        # self.model = torch.nn.DataParallel(self.model, device_ids=Config.device_ids)
        #
        self.model = self.model.to(device) # 通用的：移至GPU
        #
        # optimizer
        #
        optimizer = optim.Adam(self.model.parameters(), lr=Config.LR)
        optimizer.zero_grad()
        # 下面是分布式版本
        # optimizer = optim.Adam(self.model.parameters(), lr=Config.LR)
        # optimizer = torch.nn.DataParallel(optimizer, device_ids=Config.device_ids)  # 优化器
        # optimizer.module.zero_grad()

        # Start iteration over Training videos
        # print("self.dataset.train_videos: ", self.dataset.train_videos)
        cnt_step = 0
        with SummaryWriter(log_dir="summary/train_{0}_lr={1}_lam={2}_{3}".format(
                            Config.dataset_name, Config.LR,
                            Config.LAM, Config.normal_or_dist),
                           comment="{}".format(Config.dataset_name)) as writer:
            for epoch in range(Config.epoch):
                for cl_idx, video_id in enumerate(self.dataset.train_videos):
                    # 逐个子目录处理 from Train001~Train016
                    # Run the train
                    self.dataset.train(video_id) # 加载当前子目录的所有帧组成一个大clip到内存
                    #
                    loader = DataLoader(self.dataset,
                                        collate_fn=self.dataset.collate_fn,
                                        num_workers=Config.num_workers,
                                        batch_size=Config.batch_size,
                                        shuffle=Config.shuffle)

                    for i, (x, y) in tqdm(enumerate(loader),
                                          desc=f'Training for {self.dataset}'):
                        #
                        cnt_step = cnt_step + 1 # 一个step 一个 batch: 1380张？
                        # print("x, y .shape: ", x.shape, y.shape) # 1380, 1, 8, 32, 32
                        x = x.to(device)
                        x_r, z, z_dist = self.model(x)
                        total_loss_bp = self.loss(x, x_r, z, z_dist) # 返回的是一个tensor
                        # print("z, z_dist: ", z, z_dist)
                        reconstruction_loss = self.loss.reconstruction_loss
                        # print("reconstruction_loss: ", reconstruction_loss)
                        autoregression_loss = self.loss.autoregression_loss
                        # print("autoregression_loss: ", autoregression_loss)
                        total_loss = self.loss.total_loss
                        print("\ntotal_loss: ", total_loss) # TODO 估计还要修改
                        if cnt_step % Config.plot_every == 0:
                            writer.add_scalars("train_loss",
                                               {'total_loss':total_loss,
                                                'reconstruction_loss':reconstruction_loss,
                                                'autoregression_loss':autoregression_loss
                                                },
                                               cnt_step)
                        if cnt_step % Config.save_ckpt_every == 0:
                            # 保存模型 （在每个epoch结束时保存）# 或者根据 cnt_step设置
                            ckpt_path = '{prefix}{dataset}_{time}.pkl'.format(
                                prefix=Config.prefix,
                                dataset=Config.dataset_name,
                                time=time.strftime('%m%d_%H%M')  # 这个要和下面 save()无限近
                            )
                            torch.save(self.model.state_dict(), ckpt_path)
                            print("epoch {} complete !".format(epoch))
                        total_loss_bp.backward()
                        # 梯度裁剪
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                       max_norm=20)
                        #
                        # 保存所有变量及其梯度 [非常耗费时间，需要更优雅的方式，TODO]
                        # for name, param in self.model.named_parameters():
                        #     writer.add_histogram(name, param, cnt_step)
                        #     writer.add_histogram(name+"_grad", param.grad, cnt_step)
                        #
                        optimizer.step()
                        # optimizer.module.step() # 分布式
                        # 以及 validation ? TODO
            # 结束之前查看下graph
            # model_input = torch.rand([1380, 1, 8, 32, 32])
            # writer.add_graph(self.model, input_to_model=model_input)
            # 注意：writer.add_graph：要求pytorch.version大于 1.0.0，若0.4.1会报错

    #########################################################################################

    @torch.no_grad() # 在这里以装饰器的方法，静止back propogation，方便代码重用
    def test_video_anomaly_detection(self):
        # type: () -> None
        """
        Actually performs tests.
        """

        c, t, h, w = self.dataset.raw_shape

        # Load the checkpoint
        self.model.load_w(self.checkpoint)

        # Prepare a table to show results
        vad_table = self.empty_table

        # Set up container for novelty scores from all test videos
        global_llk = []
        global_rec = []
        global_ns = []
        global_y = []

        # Get accumulators，干嘛的？答：get frame-level scores from clip-level scores
        results_accumulator_llk = ResultsAccumulator(time_steps=t)
        results_accumulator_rec = ResultsAccumulator(time_steps=t)

        cnt_step = 0 # global_step
        with SummaryWriter(log_dir="summary/test_{0}".format(
                Config.output_file.split('.')[0]),
                comment="{}".format(Config.dataset_name)) as writer:
            # Start iteration over test videos
            for cl_idx, video_id in enumerate(self.dataset.test_videos):
                # test_videos 的内容是：TestXXX(XXX：001~012) 这些目录名，每个目录名保存有一个
                # 视频的所有帧，所以代表一个视频，即 video_id

                # Run the test
                self.dataset.test(video_id) # 设置好cur_video_frames【其实是整个视频的全部clips】，
                # cur_video_gt，cur_len【其实是clips number】
                loader = DataLoader(self.dataset,
                                    num_workers=Config.num_workers,
                                    shuffle=Config.shuffle,
                                    batch_size=Config.batch_size,
                                    collate_fn=self.dataset.collate_fn) # 临时构建loader
                # 因为是 inference，所以没有 batch_size (或者说==1)
                # collate_fn：实际作用是：TODO

                # Build score containers
                sample_llk = np.zeros(shape=(len(loader) + t - 1,))
                sample_rec = np.zeros(shape=(len(loader) + t - 1,))
                # print("len(loader): ", len(loader)) # len(self.batch_sampler)
                # 因为loader会把所有Dataset的所有item都做登记，而len(dataset) ==
                # num_frames - t + 1，即所有的clips (带overlap的)，要恢复就是：
                # len(loader) + t - 1
                # print("len(loader) + t - 1: ", len(loader) + t - 1)
                sample_y = self.dataset.load_test_sequence_gt(video_id) # (n_frames,)
                # print("len(sample_y): ", len(sample_y))
                # 事实证明：(len(loader) + t - 1) == len(sample_y), len(loader) =
                #
                for i, (x, y) in tqdm(enumerate(loader), desc=f'Computing scores for {self.dataset}'):
                    #
                    cnt_step = cnt_step + 1

                    x = x.to(device)

                    x_r, z, z_dist = self.model(x)

                    ttloss = self.loss(x, x_r, z, z_dist) # 记住，self.loss其实一个 object，这里
                    # 被执行了 forwrd()，所以等于修改了 object (即 self.loss被修改了)
                    total_loss = self.loss.total_loss
                    reconstruction_loss = self.loss.reconstruction_loss
                    autoregression_loss = self.loss.autoregression_loss
                    # write all loss
                    # if cnt_step % Config.plot_every == 0:
                    #     writer.add_scalars("test_loss",
                    #                        {'total_loss': total_loss,
                    #                         'reconstruction_loss': reconstruction_loss,
                    #                         'autoregression_loss': autoregression_loss
                    #                         },
                    #                        cnt_step)

                    # Feed results accumulators: 模仿一个队列，队尾进，队头出
                    # 我的办法：通过设置断点，进去看results_accumulator_llk是怎么工作的？
                    # 因为 batch_szie == 1, 所以push了 it(==num_clips==num_frames-t+1)次，
                    # 所以还有 (t - 1) 帧没有计算 loss，留到 下面的 while
                    results_accumulator_llk.push(self.loss.autoregression_loss)
                    results_accumulator_rec.push(self.loss.reconstruction_loss)
                    sample_llk[i] = results_accumulator_llk.get_next()
                    sample_rec[i] = results_accumulator_rec.get_next()

                # Get last results
                # 计算剩下的 (t-1)帧各自的 loss
                while results_accumulator_llk.results_left != 0:
                    index = (- results_accumulator_llk.results_left)
                    sample_llk[index] = results_accumulator_llk.get_next()
                    sample_rec[index] = results_accumulator_rec.get_next()

                min_llk, max_llk, min_rec, max_rec = self.compute_normalizing_coefficients(sample_llk, sample_rec)

                # Compute the normalized scores and novelty score
                sample_llk = normalize(sample_llk, min_llk, max_llk)
                sample_rec = normalize(sample_rec, min_rec, max_rec)
                sample_ns = novelty_score(sample_llk, sample_rec)
                # 绘制 score-map
                # print("len of sample_ns:", len(sample_ns))
                fig_novelty_score = plt.figure()
                plt.title('novelty_score of {}'.format(video_id))
                plt.plot(range(len(sample_ns)), sample_ns, color='green',
                                             label='novelty_score')
                plt.xlabel('frames')
                plt.ylabel('novelty_score')
                writer.add_figure('Novelty Score', fig_novelty_score, global_step=cl_idx)

                # Update global scores (used for global metrics)
                global_llk.append(sample_llk)
                global_rec.append(sample_rec)
                global_ns.append(sample_ns)
                global_y.append(sample_y)

                try:
                    # Compute AUROC for this video
                    this_video_metrics = [
                        roc_auc_score(sample_y, sample_llk),  # likelihood metric
                        roc_auc_score(sample_y, sample_rec),  # reconstruction metric
                        roc_auc_score(sample_y, sample_ns)    # novelty score
                    ]
                    vad_table.add_row([video_id] + this_video_metrics)
                except ValueError:
                    # This happens for sequences in which all frames are abnormal
                    # Skipping this row in the table (the sequence will still count for global metrics)
                    continue

            # Compute global AUROC and print table
            global_llk = np.concatenate(global_llk)
            global_rec = np.concatenate(global_rec)
            global_ns = np.concatenate(global_ns)
            global_y = np.concatenate(global_y)
            global_metrics = [
                roc_auc_score(global_y, global_llk),  # likelihood metric
                roc_auc_score(global_y, global_rec),  # reconstruction metric
                roc_auc_score(global_y, global_ns)    # novelty score
            ]
            vad_table.add_row(['avg'] + list(global_metrics))
            print(vad_table)

            # Save table
            with open(self.output_file, mode='w') as f:
                f.write(str(vad_table))
            #
            # 查看下网络
            # model_input = torch.rand([1380, 1, 8, 32, 32])
            # writer.add_graph(self.model, input_to_model=model_input)


    @staticmethod
    def compute_normalizing_coefficients(sample_llk, sample_rec):
        # type: (np.ndarray, np.ndarray) -> Tuple[float, float, float, float]
        """
        Computes normalizing coefficients for the computationof the novelty score.

        :param sample_llk: array of log-likelihood scores.
        :param sample_rec: array of reconstruction scores.
        :return: a tuple of normalizing coefficients in the form (llk_min, llk_max, rec_min, rec_max).
        """
        return sample_llk.min(), sample_llk.max(), sample_rec.min(), sample_rec.max()

    @property
    def empty_table(self):
        # type: () -> PrettyTable
        """
        Sets up a nice ascii-art table to hold results.
        This table is suitable for the video anomaly detection setting.
        :return: table to be filled with auroc metrics.
        """
        table = PrettyTable()
        table.field_names = ['VIDEO-ID', 'AUROC-LLK', 'AUROC-REC', 'AUROC-NS']
        table.float_format = '0.3'
        return table
