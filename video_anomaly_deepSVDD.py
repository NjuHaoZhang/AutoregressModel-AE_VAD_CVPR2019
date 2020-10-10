from typing import Tuple

import numpy as np
import torch
from prettytable import PrettyTable
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.base import VideoAnomalyDetectionDataset
from models.base import BaseModule
from models.loss_functions import LSALoss,LSALoss_deepSVDD
from utils import normalize
from utils import novelty_score
#
from torch import optim # 优化器 by HaoZhang
from tensorboardX import SummaryWriter
import os,time
import matplotlib.pyplot as plt
plt.switch_backend('agg')
#
import utils
import logging



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


class VideoAnomalyDetectionResultHelper_deepSVDD(object):
    """
    Performs tests for video anomaly detection datasets (UCSD Ped2 or Shanghaitech).
    """

    def __init__(self, dataset_train, dataset_eval, model, params):
        # type: (VideoAnomalyDetectionDataset, BaseModule, str, str) -> None
        """
        Class constructor.

        :param dataset: dataset class.
        :param model: pytorch model to evaluate.
        :param checkpoint: path of the checkpoint for the model.
        :param output_file: text file where to save results.
        """
        #
        self.dataset_train, self.dataset_eval = dataset_train, dataset_eval
        self.model = model
        self.device = torch.device("cuda:" + params.device)  # 配置使用的GPU
        self.params = params
        #print("self.params.restore_file: ", self.params.restore_file)
        # Set up loss function
        # for AE Training & c_calculating
        #c_tmp = torch.zeros([690, 2 * (self.params.code_length)], device=self.device)
        # for params_opt
        self.c = torch.zeros([690, 2 * (self.params.code_length)], device=self.device) # for ped2 , ped1 and other dataset 再说
        self.loss = LSALoss_deepSVDD(lam_rec=self.params.lam_rec, lam_svdd=self.params.lam_svdd,
                                     c=self.c, R=self.params.R, nu=self.params.nu,
                                     objective=self.params.objective)

    #######################################################################################
    # by HaoZhang
    # 下面用于超参搜索

    def hyperparas_search(self):

        # Set the logger
        utils.set_logger(os.path.join(self.params.model_dir, 'params_opt_train.log'))
        # Create the input data pipeline
        logging.info("Loading the datasets...")

        # fetch dataloaders
        # train_dl = self.dataset_train
        # val_dl = self.dataset_eval

        logging.info("- done.")

        # Define the model and optimizer
        # 加载最原始论文提供的ckpt，Load the checkpoint
        checkpoint = self.params.restore_file
        # if os.path.exists(checkpoint):
        #     print("{} load !".format(checkpoint))
        #     self.ckpt = torch.load(checkpoint)
        #     self.model.load_state_dict(self.ckpt['net_dict'])
        if checkpoint is not None: # TODO 有bug要修
            restore_path = checkpoint # checkpoints/ped2/code_length_128.pth.tar
            print("restore_path: ", restore_path)
            logging.info("Restoring parameters from {}".format(restore_path))
            utils.load_checkpoint(restore_path, self.model)
        self.model = self.model.to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.params.LR)

        # fetch loss function and metrics
        self.c = load_init_center_c(self.params.dataset_name, self.params.code_length).to(self.device)
        self.loss = LSALoss_deepSVDD(lam_rec=self.params.lam_rec, lam_svdd=self.params.lam_svdd,
                                     c=self.c, R=self.params.R, nu=self.params.nu,
                                     objective=self.params.objective)
        loss_fn = self.loss
        metrics = utils.metrics

        # Train the model
        logging.info("Starting training for {} epoch(s)".format(self.params.epoch))
        # restore_file = self.params.restore_file_path # 默认为None, TODO
        restore_file = None
        self.train_and_evaluate(self.model, self.dataset_train, self.dataset_eval,
                                optimizer, loss_fn, metrics, self.params, self.params.model_dir,
                                restore_file)

    def train_and_evaluate(self, model, train_dataloader, val_dataloader,
                           optimizer, loss_fn, metrics, params, model_dir,
                            restore_file=None):
        # 加载现在idea产生的ckpt
        # reload weights from restore_file if specified
        # if restore_file is not None: # TODO 有bug要修
        #     restore_path = os.path.join(model_dir, restore_file + '.pth.tar')
        #     logging.info("Restoring parameters from {}".format(restore_path))
        #     utils.load_checkpoint(restore_path, model, optimizer)

        best_val_acc = 0.0

        cnt_step = 0
        for epoch in range(params.epoch):

            # Run one epoch
            logging.info("Epoch {}/{}".format(epoch + 1, params.epoch))

            # 主要是看下这里model是否共享，即train好的model是否给了test，
            # 下一个epoch的model是否来自上一个epoch train的 model
            logging.info("model id before train: ", id(model))

            # compute number of batches in one epoch (one full pass over the training set)
            self.train_joint_for_paramsopt(model, optimizer, loss_fn, train_dataloader, metrics, params)

            #
            logging.info("model id before test: ", id(model))

            # Evaluate for one epoch on validation set
            val_metrics = self.test_joint_for_paramsopt(model, loss_fn, val_dataloader, metrics, params)

            val_acc = val_metrics['auc']
            is_best = val_acc >= best_val_acc

            # Save weights
            utils.save_checkpoint({'epoch': epoch + 1,
                                   'state_dict': model.state_dict(),
                                   'optim_dict': optimizer.state_dict(),
                                   'c': self.c,
                                   'R': self.params.R,
                                   'nu': self.params.nu,
                                   'lam_rec': self.params.lam_rec,
                                   'lam_svdd': self.params.lam_svdd
                                   },
                                  is_best=is_best,
                                  checkpoint=model_dir)

            # If best_eval, best_save_path
            if is_best:
                logging.info("- Found new best accuracy")
                best_val_acc = val_acc

                # Save best val metrics in a json file in the model directory
                best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
                utils.save_dict_to_json(val_metrics, best_json_path)

            # Save latest val metrics in a json file in the model directory
            last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
            utils.save_dict_to_json(val_metrics, last_json_path)

    def train_joint_for_paramsopt(self,
                            model, optimizer, loss_fn, train_dataloader, metrics, params):
        c, t, h, w = train_dataloader.raw_shape
        #
        model.train()    # set model to training mode
        optimizer.zero_grad() # clear previous gradients
        #
        cnt_step = 0
        # summary for current training loop and a running average object for loss
        summ = []
        # loss_avg = utils.RunningAverage()
        with SummaryWriter(log_dir="summary/train_deepSVDD/train_{0}_lr={1}_lam_rec={2}_"
                                   "lam_svdd={3}_code_length={4}".format(
                    self.params.dataset_name, self.params.LR, self.params.lam_rec, self.params.lam_svdd,
                    self.params.code_length), comment="{}".format(self.params.dataset_name)) as writer:
            for cl_idx, video_id in enumerate(train_dataloader.train_videos):
                # 逐个子目录处理 from Train001~Train016
                # Run the train
                train_dataloader.train(video_id)  # 加载当前子目录的所有帧组成一个大clip到内存
                #
                loader = DataLoader(train_dataloader,
                                    collate_fn=train_dataloader.collate_fn,
                                    num_workers=self.params.num_workers,
                                    batch_size=self.params.batch_size,
                                    shuffle=True)

                for i, (x, y) in tqdm(enumerate(loader),
                                      desc=f'Training for {self.params.dataset_name}'):
                    #
                    cnt_step = cnt_step + 1  # 一个step 一个 batch: 1380张
                    # print("x, y .shape: ", x.shape, y.shape) # 1380, 1, 8, 32, 32
                    x = x.to(self.device)
                    x_r, z = model(x)
                    # print("z.size: ", z.size())
                    # print("x_r.size: ", x_r.size())
                    #

                    z = z.view(-1, 690,2*(self.params.code_length))
                    # print("z.size: ", z.size())
                    #
                    dist = torch.sum((z - self.c) ** 2, dim=1)  # points to center
                    #
                    total_loss_bp = loss_fn(x, x_r, z)  # 返回的是一个tensor
                    # print("z, z_dist: ", z, z_dist)0
                    reconstruction_loss = loss_fn.reconstruction_loss
                    # print("reconstruction_loss: ", reconstruction_loss)
                    deepSVDD_loss = loss_fn.deepSVDD_loss
                    # print("deepSVDD_loss: ", deepSVDD_loss)
                    total_loss = loss_fn.total_loss
                    print("\ntotal_loss: ", total_loss)
                    if cnt_step % params.plot_every == 0:
                        writer.add_scalars("train_loss",
                                           {'total_loss': total_loss,
                                            'reconstruction_loss': reconstruction_loss,
                                            'deepSVDD_loss': deepSVDD_loss
                                            },
                                           cnt_step)
                    # if cnt_step % params.save_ckpt_every == 0:
                    #     # 保存模型 （在每个epoch结束时保存）# 或者根据 cnt_step设置
                    #     ckpt_path = '{prefix}{dataset}_{time}.pkl'.format(
                    #         prefix=Config.prefix,
                    #         dataset=Config.dataset_name,
                    #         time=time.strftime('%m%d_%H%M')  # 这个要和下面 save()无限近
                    #     )
                    #     net_dict = self.model.state_dict()
                    #     #
                    #     torch.save({'R': self.R,
                    #                 'c': self.c,
                    #                 'net_dict': net_dict, }, ckpt_path)
                    #     print("epoch {} complete !".format(epoch))
                    total_loss_bp.backward() # 确保 optimizer.zero_grad()
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   max_norm=20)
                    optimizer.step()
                    # Update hypersphere radius R on mini-batch distances
                    # if (self.objective == 'soft-boundary') and (epoch >= params.warm_up_n_steps):
                    #     self.R.data = torch.tensor(get_radius(dist, self.nu), device=self.device)
                    #
                    # 由于没有 label，无法计算acc，所以直接打印loss
                    # Evaluate summaries only once in a while
                    if i % params.save_summary_steps == 0:
                        summary_batch = {}
                        summary_batch['loss'] = total_loss
                        summ.append(summary_batch)
                # compute mean of all metrics in summary
                metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
                metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
                logging.info("- Train metrics: " + metrics_string)


    @torch.no_grad()  # 在这里以装饰器的方法，静止back propogation，方便代码重用
    def test_joint_for_paramsopt(self, model, loss_fn, val_dataloader, metrics, params):
        # type: () -> None
        """
        Actually performs tests.
        """
        c, t, h, w = val_dataloader.raw_shape

        # set model to evaluation mode
        model.eval()

        # summary for current eval loop
        # summ = []
        metrics = {}

        # Load the checkpoint
        # self.model.load_w(self.checkpoint)
        # self.ckpt = torch.load(self.checkpoint)
        # self.model.load_state_dict(self.ckpt['net_dict'])
        # self.R = self.ckpt['R']
        # self.c = self.ckpt['c']

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

        cnt_step = 0  # global_step
        with SummaryWriter(log_dir="summary/test_deepSVDD/test_{0}_lr={1}_lam_rec={2}_"
                                   "lam_svdd={3}_code_length={4}".format(
                    self.params.dataset_name, self.params.LR, self.params.lam_rec, self.params.lam_svdd,
                    self.params.code_length), comment="{}".format(self.params.dataset_name)) as writer:
            # Start iteration over test videos
            for cl_idx, video_id in enumerate(val_dataloader.test_videos):
                # test_videos 的内容是：TestXXX(XXX：001~012) 这些目录名，每个目录名保存有一个
                # 视频的所有帧，所以代表一个视频，即 video_id

                # Run the test
                val_dataloader.test(video_id)  # 设置好cur_video_frames【其实是整个视频的全部clips】，
                # cur_video_gt，cur_len【其实是clips number】
                loader = DataLoader(val_dataloader,
                                    num_workers=1,
                                    shuffle=False,
                                    batch_size=1,
                                    collate_fn=val_dataloader.collate_fn)  # 临时构建loader
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
                sample_y = val_dataloader.load_test_sequence_gt(video_id)  # (n_frames,)
                # print("len(sample_y): ", len(sample_y))
                # 事实证明：(len(loader) + t - 1) == len(sample_y), len(loader) =
                #
                for i, (x, y) in tqdm(enumerate(loader),
                                      desc=f'Computing scores for {self.params.dataset_name}'):
                    #
                    cnt_step = cnt_step + 1

                    x = x.to(self.device)

                    x_r, z = model(x)
                    z = z.view(-1, 690, 2*(self.params.code_length))
                    # print("in 327 line, z.size: ", z.size())

                    ttloss = loss_fn(x, x_r, z)  # 记住，self.loss其实一个 object，这里
                    # 被执行了 forwrd()，所以等于修改了 object (即 self.loss被修改了)
                    total_loss = loss_fn.total_loss
                    reconstruction_loss = loss_fn.reconstruction_loss
                    deepSVDD_loss = loss_fn.deepSVDD_loss
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
                    results_accumulator_llk.push(loss_fn.deepSVDD_loss)
                    results_accumulator_rec.push(loss_fn.reconstruction_loss)
                    sample_llk[i] = results_accumulator_llk.get_next()
                    sample_rec[i] = results_accumulator_rec.get_next()

                # Get last results
                # 计算剩下的 (t-1)帧各自的 loss
                while results_accumulator_llk.results_left != 0:
                    index = (- results_accumulator_llk.results_left)
                    sample_llk[index] = results_accumulator_llk.get_next()
                    sample_rec[index] = results_accumulator_rec.get_next()

                min_llk, max_llk, min_rec, max_rec = self.compute_normalizing_coefficients(
                    sample_llk, sample_rec)

                # Compute the normalized scores and novelty score
                sample_llk = normalize(sample_llk, min_llk, max_llk)
                sample_rec = normalize(sample_rec, min_rec, max_rec)
                sample_ns = novelty_score(sample_llk, sample_rec)
                # # 绘制 score-map
                # # print("len of sample_ns:", len(sample_ns))
                # fig_novelty_score = plt.figure()
                # plt.title('novelty_score of {}'.format(video_id))
                # plt.plot(range(len(sample_ns)), sample_ns, color='green',
                #          label='novelty_score')
                # plt.xlabel('frames')
                # plt.ylabel('novelty_score')
                # writer.add_figure('Novelty Score', fig_novelty_score, global_step=cl_idx)

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
                        roc_auc_score(sample_y, sample_ns)  # novelty score
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
                roc_auc_score(global_y, global_ns)  # novelty score
            ]
            vad_table.add_row(['avg'] + list(global_metrics))
            print(vad_table)

            # # Save table
            # with open(self.output_file, mode='w') as f:
            #     f.write(str(vad_table))
            #     #
            #     # 查看下网络
            #     # model_input = torch.rand([1380, 1, 8, 32, 32])
            #     # writer.add_graph(self.model, input_to_model=model_input)
            # print("ag_auc: ", list(global_metrics)[2])
            metrics['auc'] = list(global_metrics)[2]
            return  metrics# 返回 avg_auc

    ########################################################################################
    # BY haozhang
    # 用于计算质心和测试下训练和测试code

    def train_video_anomaly_detection(self):
        # type: () -> None
        """
        训练
        """
        # self.train_AE() # 训练的第一阶段：AE, 然后 self中的model, c, R, nu,
        # 直接用在下面函数
        #print("self.params.restore_file: ", self.params.restore_file)
        self.calcu_paras() # 利用第一阶段的Net 计算c,
        # 上面两步还是丢到其他预处理步骤，不妨到这个helper里面？？？ TODO
        # self.train_deepSVDD()  # 训练的第二阶段：deepSVDD, 这一步现在在我的idea里面不需要，直接开始调参

    def train_AE(self):

        import inspect
        # from gpu_men_track import MemTracker  # 引用显存跟踪代码
        # frame = inspect.currentframe()
        # gpu_tracker = MemTracker(frame)  # 创建显存检测对象

        # Set the logger
        utils.set_logger(os.path.join(self.params.log_root_path, 'train_AE.log'))

        c, t, h, w = self.dataset_train.raw_shape

        # Load the checkpoint
        if os.path.exists(self.params.model_ckpt):
            print("{} load !".format(self.params.model_ckpt))
            self.ckpt = torch.load(self.params.model_ckpt)
            # 重新解析
            self.model.load_state_dict(self.ckpt['net_dict'])
            self.R = self.ckpt['R']
            self.c = self.ckpt['c']
            #
        # 下面是分布式版本
        # self.model = torch.nn.DataParallel(self.model, device_ids=Config.device_ids)
        #
        self.model = self.model.to(self.device)  # 通用的：移至GPU
        # gpu_tracker.track()  # 开始检测

        self.model.train()  # 设置为 train mode
        # gpu_tracker.track()  # 开始检测

        #
        # optimizer
        #
        optimizer = optim.Adam(self.model.parameters(), lr=self.params.LR)
        optimizer.zero_grad()
        # 下面是分布式版本
        # optimizer = optim.Adam(self.model.parameters(), lr=Config.LR)
        # optimizer = torch.nn.DataParallel(optimizer, device_ids=Config.device_ids)  # 优化器
        # optimizer.module.zero_grad()

        # Start iteration over Training videos
        # print("self.dataset.train_videos: ", self.dataset.train_videos)
        cnt_step = 0
        with SummaryWriter(log_dir="summary/train_AE_{0}_lr={1}_lam_rec={2}_lam_svdd={3}"
                                   "_code_length={4}".format(
            self.params.dataset_name, self.params.LR,
            self.params.lam_rec, self.params.lam_svdd, self.params.code_length),
                comment="{}".format(self.params.dataset_name)) as writer:
            for epoch in range(self.params.epoch):
                for cl_idx, video_id in enumerate(self.dataset_train.train_videos):
                    # 逐个子目录处理 from Train001~Train016
                    # Run the train
                    self.dataset_train.train(video_id)  # 加载当前子目录的所有帧组成一个大clip到内存
                    #
                    loader = DataLoader(self.dataset_train,
                                        collate_fn=self.dataset_train.collate_fn,
                                        num_workers=self.params.num_workers,
                                        batch_size=self.params.batch_size,
                                        shuffle=True)
                    # gpu_tracker.track()  # 开始检测

                    for i, (x, y) in tqdm(enumerate(loader),
                                          desc=f'Training for {self.params.dataset_name}'):
                        #
                        cnt_step = cnt_step + 1  # 一个step 一个 batch: 1380张
                        # print("x, y .shape: ", x.shape, y.shape) # 1380, 1, 8, 32, 32

                        # if hasattr(torch.cuda, 'empty_cache'):
                        # torch.cuda.empty_cache()

                        x = x.to(self.device)
                        # gpu_tracker.track()  # 开始检测
                        x_r, z = self.model(x)
                        # gpu_tracker.track()  # 开始检测
                        print("z.size: ", z.size())
                        # print("x_r.size: ", x_r.size())
                        #

                        z = z.view(-1, 690, 2*(self.params.code_length)) # 690 个 patch
                        print("z.size: ", z.size())
                        # print("z.size: ", z.size())
                        total_loss_bp = self.loss(x, x_r, z)  # 返回的是一个tensor
                        # print("z, z_dist: ", z, z_dist)0
                        reconstruction_loss = self.loss.reconstruction_loss
                        # print("reconstruction_loss: ", reconstruction_loss)
                        deepSVDD_loss = self.loss.deepSVDD_loss
                        # print("deepSVDD_loss: ", deepSVDD_loss)
                        total_loss = self.loss.total_loss
                        del x, x_r, z
                        logging.info("total_loss: {}".format(total_loss))
                        if cnt_step % (self.params.plot_every) == 0:
                            writer.add_scalars("train_loss",
                                               {'total_loss': total_loss,
                                                'reconstruction_loss': reconstruction_loss,
                                                'deepSVDD_loss': deepSVDD_loss
                                                },
                                               cnt_step)
                        # Save weights
                        if cnt_step % (self.params.save_ckpt_every) == 0:
                            utils.save_checkpoint_for_Train({'epoch': epoch + 1,
                                                   'state_dict': self.model.state_dict(),
                                                   'optim_dict': optimizer.state_dict() },
                                                  self.params.model_save_dir,
                                                  self.params.code_length)
                        total_loss_bp.backward()
                        # 梯度裁剪
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                       max_norm=20)
                        optimizer.step()

    def calcu_paras(self):
        # 计算 c
        import os, time
        from datasets import UCSDPed1_deepSVDD_TRAIN

        dataset = UCSDPed1_deepSVDD_TRAIN(path="data/UCSD_Anomaly_Dataset.v1p2")
        # net = self.model # 直接使用上一步 AE 的 pretrin model
        # if os.path.exists(self.checkpoint):
        #     print("{} load !".format(self.checkpoint))
        #     self.ckpt = torch.load(self.checkpoint)
        #     self.model.load_state_dict(self.ckpt['net_dict'])
        # print("self.params", self.params)
        # print("self.params.restore_file: ", self.params.restore_file)
        if self.params.restore_file is not None: # TODO 有bug要修
            restore_path = self.params.restore_file # checkpoints/ped2/code_length_128.pth.tar
            print("restore_path: ", restore_path)
            logging.info("Restoring parameters from {}".format(restore_path))
            utils.load_checkpoint(restore_path, self.model)

            #
            # import re
            # code_length = re.findall(r"\d+", restore_path)
            batch_size = 2
            c = init_center_c(dataset, self.model, batch_size, self.device, self.params.code_length)
            torch.save(c, "c_init_ped1_{}.pt".format(self.params.code_length))
            cl = torch.load("c_init_ped1_{}.pt".format(self.params.code_length))
            print("c1.shape: ", cl.shape)

        # self.c = c # 赋给当前 helper

    def train_deepSVDD(self):

        '''
        # DeepSVDD params & DNN params
        assert params.objective in ('one-class', 'soft-boundary'), \
            "Objective must be either 'one-class' or 'soft-boundary'."
        self.objective = params.objective
        # if c then ... TODO
        c = load_init_center_c(params.dataset_name) # init c
        self.R = torch.tensor(params.R, device=self.device)
        # radius R initialized with 0 by default.
        self.c = torch.tensor(c, device=self.device) \
            if c is not None else None
        self.nu = params.nu
        :return:
        '''

        # Set the logger
        utils.set_logger(os.path.join(self.model_dir, 'train_deepSVDD.log'))
        # Create the input data pipeline
        # logging.info("Loading the datasets...")
        # logging.info("- done.")

        self.model = self.model.to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.LR)

        # reload weights from restore_file if specified
        if self.restore_file is not None:  # 真正训练要用这个
            restore_path = os.path.join(self.model_dir, self.restore_file + '.pth.tar')
            logging.info("Restoring parameters from {}".format(restore_path))
            utils.load_checkpoint(restore_path, self.model, optimizer)
        elif os.path.exists(self.checkpoint): # 加载CVPR2019的best_model 方案
            print("{} load !".format(self.checkpoint))
            self.ckpt = torch.load(self.checkpoint)
            self.model.load_state_dict(self.ckpt['net_dict'])
            # 这个checkpoint没有c,R, nu, lam_xxx 可以加载，因为是加载别人论文的ckpt
        else:
            pass

        c, t, h, w = self.dataset_train.raw_shape
        self.model.train()  # set model to training mode
        optimizer.zero_grad()  # clear previous gradients

        cnt_step = 0
        with SummaryWriter(log_dir="summary/train_deepSVDD/train_{0}_lr={1}_lam_rec={2}"
                                   "_lam_svdd={3}_nu={4}_{5}_{6}".format(
                self.dataset_name, self.LR, self.lam_rec, self.lam_svdd, self.nu,
                self.params.normal_or_dist, time.strftime('%m%d_%H%M')),
                comment="{}".format(self.dataset_name)) as writer:
            # Train the model
            logging.info("Starting training for {} epoch(s)".format(self.epoch))
            for epoch in range(self.epoch):
                for cl_idx, video_id in enumerate(self.dataset_train.train_videos):
                    # 逐个子目录处理 from Train001~Train016
                    # Run the train
                    self.dataset_train.train(video_id)  # 当前子目录的所有帧组成一个大clip
                    #
                    loader = DataLoader(self.dataset_train,
                                        collate_fn=self.dataset_train.collate_fn,
                                        num_workers=self.num_workers,
                                        batch_size=self.batch_size,
                                        shuffle=True)

                    for i, (x, y) in tqdm(enumerate(loader),
                                          desc=f'Training for {self.dataset_name}'):
                        #
                        cnt_step = cnt_step + 1  # 一个step 一个 batch: 1380张
                        # print("x, y .shape: ", x.shape, y.shape) # 1380, 1, 8, 32, 32
                        x = x.to(self.device)
                        x_r, z = self.model(x)
                        #
                        z = z.view(-1, 690, 128)
                        # print("z.size: ", z.size())
                        #
                        dist = torch.sum((z - self.c) ** 2, dim=1)  # points to center
                        #
                        total_loss_bp = self.loss(x, x_r, z)  # 返回的是一个tensor
                        # print("z, z_dist: ", z, z_dist)0
                        reconstruction_loss = self.loss.reconstruction_loss
                        # print("reconstruction_loss: ", reconstruction_loss)
                        deepSVDD_loss = self.loss.deepSVDD_loss
                        # print("deepSVDD_loss: ", deepSVDD_loss)
                        total_loss = self.loss.total_loss
                        print("\ntotal_loss: ", total_loss)
                        if cnt_step % self.params.plot_every == 0:
                            writer.add_scalars("train_loss",
                                               {'total_loss': total_loss,
                                                'reconstruction_loss': reconstruction_loss,
                                                'deepSVDD_loss': deepSVDD_loss
                                                },
                                               cnt_step)

                        total_loss_bp.backward()  # 确保 optimizer.zero_grad()
                        # 梯度裁剪
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                       max_norm=20)
                        optimizer.step()

                        if cnt_step % self.params.save_ckpt_every == 0:
                            # Save weights
                            utils.save_checkpoint({'epoch': epoch + 1,
                                   'state_dict': self.model.state_dict(),
                                   'optim_dict': optimizer.state_dict(),
                                   'c': self.c,
                                   'R': self.R,
                                   'nu': self.nu,
                                   'lam_rec': self.lam_rec,
                                   'lam_svdd': self.lam_svdd
                                   },
                                  is_best=True,
                                  checkpoint=self.model_dir)

    @torch.no_grad() # 在这里以装饰器的方法，静止back propogation，方便代码重用
    def test_video_anomaly_detection(self):
        # type: () -> None
        """
        Actually performs tests.
        """

        c, t, h, w = self.dataset.raw_shape

        # Load the checkpoint
        # self.model.load_w(self.checkpoint)
        self.ckpt = torch.load(self.checkpoint)
        self.model.load_state_dict(self.ckpt['net_dict'])
        self.R = self.ckpt['R']
        self.c = self.ckpt['c']

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

                    x_r, z = self.model(x)
                    z = z.view(-1, 690, 128)  # ?个clip，每个clip有690个patch，每个patch的embedding为128-dim vector
                    print("in 327 line, z.size: ", z.size())


                    ttloss = self.loss(x, x_r, z) # 记住，self.loss其实一个 object，这里
                    # 被执行了 forwrd()，所以等于修改了 object (即 self.loss被修改了)
                    total_loss = self.loss.total_loss
                    reconstruction_loss = self.loss.reconstruction_loss
                    deepSVDD_loss = self.loss.deepSVDD_loss
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
                    results_accumulator_llk.push(self.loss.deepSVDD_loss)
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

    ###############################################

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

   ########################################################################################

# 一些辅助函数，后面放到 utils中去
def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)

def load_init_center_c(dataset_name="ucsd_ped2", code_length=64):
    if dataset_name == "ucsd_ped2":
        return torch.load("c_init_ped2_{}.pt".format(str(code_length)))
    if dataset_name == "ucsd_ped1":
        return torch.load("c_init_ped1_{}.pt".format(str(code_length)))

# init c
def init_center_c(dataset, net, batch_size, device, code_length, eps=0.1):
    """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    n_samples = 0
    c = torch.zeros([690,2*code_length], device=device) # for ped2, 其他数据集再具体处理
    net = net.to(device)
    net.eval()

    # print("(dataset.train_videos: ", (dataset.train_videos))
    for cl_idx, video_id in enumerate(dataset.train_videos):
        # print("cl_idex: ", cl_idx)
        # Run the test
        dataset.train(video_id)
        loader = DataLoader(dataset,
                            collate_fn=dataset.collate_fn,
                            num_workers=4,
                            batch_size=batch_size, # 最大能支持的batch_size
                            shuffle=False)
        #
        with torch.no_grad():
            for i, (x, y) in tqdm(enumerate(loader),desc="get_c_init of ucsd"):
                #
                x = x.to(device)

                x_r, z = net(x) # z就是我需要的 latent vector, (batchsize, seq_len, out_features)

                # print("z.shape: ", z.shape, z.shape[0]) # (1380,2,64)
                # 事实是：每个clip被处理为 690 个patch,
                # 每个32x32x1的patch的embedding 是(2,64) 的tensor，
                # 因为这里batch_size=2,所以就是2个clip，即 2*690 = 1380 个 patch
                z = z.view(-1, 690, 2*code_length) # ?个clip，每个clip有690个patch，每个patch的embedding为128-dim vector
                print("z.size: ", z.size()) # [2, 88320] # for ped2, 16 frames: 88320 = 690 * 2 * 64
                n_samples += z.shape[0]
                c += torch.sum(z, dim=0)

    c /= n_samples

    # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps

    return c