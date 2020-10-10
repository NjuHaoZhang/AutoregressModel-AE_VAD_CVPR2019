# def train_and_evaluate(self, model, train_dataloader, val_dataloader,
    #                        optimizer, loss_fn, metrics, params, model_dir,
    #                         restore_file=None):
    #     # reload weights from restore_file if specified
    #     if restore_file is not None: # TODO 有bug要修
    #         restore_path = os.path.join(model_dir, restore_file + '.pth.tar')
    #         logging.info("Restoring parameters from {}".format(restore_path))
    #         utils.load_checkpoint(restore_path, model, optimizer)
    #
    #     best_val_acc = 0.0
    #
    #     cnt_step = 0
    #     for epoch in range(params.epoch):
    #
    #         # Run one epoch
    #         logging.info("Epoch {}/{}".format(epoch + 1, params.epoch))
    #
    #         # 主要是看下这里model是否共享，即train好的model是否给了test，
    #         # 下一个epoch的model是否来自上一个epoch train的 model
    #         logging.info("model id before train: ", id(model))
    #
    #         # compute number of batches in one epoch (one full pass over the training set)
    #         self.train_joint_AE_deepSVDD(model, optimizer, loss_fn, train_dataloader, metrics, params)
    #
    #         #
    #         logging.info("model id before test: ", id(model))
    #
    #         # Evaluate for one epoch on validation set
    #         val_metrics = self.test_joint_AE_deepSVDD(model, loss_fn, val_dataloader, metrics, params)
    #
    #         val_acc = val_metrics['auc']
    #         is_best = val_acc >= best_val_acc
    #
    #         # Save weights
    #         utils.save_checkpoint({'epoch': epoch + 1,
    #                                'state_dict': model.state_dict(),
    #                                'optim_dict': optimizer.state_dict(),
    #                                'c': self.c,
    #                                'R': self.R,
    #                                'nu': self.nu,
    #                                'lam_rec': self.lam_rec,
    #                                'lam_svdd': self.lam_svdd
    #                                },
    #                               is_best=is_best,
    #                               checkpoint=model_dir)
    #
    #         # If best_eval, best_save_path
    #         if is_best:
    #             logging.info("- Found new best accuracy")
    #             best_val_acc = val_acc
    #
    #             # Save best val metrics in a json file in the model directory
    #             best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
    #             utils.save_dict_to_json(val_metrics, best_json_path)
    #
    #         # Save latest val metrics in a json file in the model directory
    #         last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
    #         utils.save_dict_to_json(val_metrics, last_json_path)
    #
    # def train_joint_AE_deepSVDD(self,
    #                             model, optimizer, loss_fn, train_dataloader, metrics, params):
    #     c, t, h, w = train_dataloader.raw_shape
    #     #
    #     model.train()    # set model to training mode
    #     optimizer.zero_grad() # clear previous gradients
    #     #
    #     cnt_step = 0
    #     # summary for current training loop and a running average object for loss
    #     summ = []
    #     # loss_avg = utils.RunningAverage()
    #     with SummaryWriter(log_dir="summary/train_deepSVDD/train_{0}_lr={1}_lam_rec={2}_"
    #                                "lam_svdd={3}_nu={4}_{5}_{6}".format(
    #                 self.dataset_name, self.LR, self.lam_rec, self.lam_svdd, self.nu,
    #                 params.normal_or_dist, time.strftime('%m%d_%H%M')),
    #                 comment="{}".format(self.dataset_name)) as writer:
    #         for cl_idx, video_id in enumerate(train_dataloader.train_videos):
    #             # 逐个子目录处理 from Train001~Train016
    #             # Run the train
    #             train_dataloader.train(video_id)  # 加载当前子目录的所有帧组成一个大clip到内存
    #             #
    #             loader = DataLoader(train_dataloader,
    #                                 collate_fn=train_dataloader.collate_fn,
    #                                 num_workers=self.num_workers,
    #                                 batch_size=self.batch_size,
    #                                 shuffle=True)
    #
    #             for i, (x, y) in tqdm(enumerate(loader),
    #                                   desc=f'Training for {self.dataset_name}'):
    #                 #
    #                 cnt_step = cnt_step + 1  # 一个step 一个 batch: 1380张
    #                 # print("x, y .shape: ", x.shape, y.shape) # 1380, 1, 8, 32, 32
    #                 x = x.to(self.device)
    #                 x_r, z = model(x)
    #                 #
    #                 z = z.view(-1, 690,128)
    #                 # print("z.size: ", z.size())
    #                 #
    #                 dist = torch.sum((z - self.c) ** 2, dim=1)  # points to center
    #                 #
    #                 total_loss_bp = loss_fn(x, x_r, z)  # 返回的是一个tensor
    #                 # print("z, z_dist: ", z, z_dist)0
    #                 reconstruction_loss = loss_fn.reconstruction_loss
    #                 # print("reconstruction_loss: ", reconstruction_loss)
    #                 deepSVDD_loss = loss_fn.deepSVDD_loss
    #                 # print("deepSVDD_loss: ", deepSVDD_loss)
    #                 total_loss = loss_fn.total_loss
    #                 print("\ntotal_loss: ", total_loss)
    #                 if cnt_step % params.plot_every == 0:
    #                     writer.add_scalars("train_loss",
    #                                        {'total_loss': total_loss,
    #                                         'reconstruction_loss': reconstruction_loss,
    #                                         'deepSVDD_loss': deepSVDD_loss
    #                                         },
    #                                        cnt_step)
    #                 # if cnt_step % params.save_ckpt_every == 0:
    #                 #     # 保存模型 （在每个epoch结束时保存）# 或者根据 cnt_step设置
    #                 #     ckpt_path = '{prefix}{dataset}_{time}.pkl'.format(
    #                 #         prefix=Config.prefix,
    #                 #         dataset=Config.dataset_name,
    #                 #         time=time.strftime('%m%d_%H%M')  # 这个要和下面 save()无限近
    #                 #     )
    #                 #     net_dict = self.model.state_dict()
    #                 #     #
    #                 #     torch.save({'R': self.R,
    #                 #                 'c': self.c,
    #                 #                 'net_dict': net_dict, }, ckpt_path)
    #                 #     print("epoch {} complete !".format(epoch))
    #                 total_loss_bp.backward() # 确保 optimizer.zero_grad()
    #                 # 梯度裁剪
    #                 torch.nn.utils.clip_grad_norm_(model.parameters(),
    #                                                max_norm=20)
    #                 optimizer.step()
    #                 # Update hypersphere radius R on mini-batch distances
    #                 # if (self.objective == 'soft-boundary') and (epoch >= params.warm_up_n_steps):
    #                 #     self.R.data = torch.tensor(get_radius(dist, self.nu), device=self.device)
    #                 #
    #                 # 由于没有 label，无法计算acc，所以直接打印loss
    #                 # Evaluate summaries only once in a while
    #                 if i % params.save_summary_steps == 0:
    #                     summary_batch = {}
    #                     summary_batch['loss'] = total_loss
    #                     summ.append(summary_batch)
    #             # compute mean of all metrics in summary
    #             metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    #             metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    #             logging.info("- Train metrics: " + metrics_string)
    #
    # @torch.no_grad()  # 在这里以装饰器的方法，静止back propogation，方便代码重用
    # def test_joint_AE_deepSVDD(self, model, loss_fn, val_dataloader, metrics, params):
    #     # type: () -> None
    #     """
    #     Actually performs tests.
    #     """
    #     c, t, h, w = val_dataloader.raw_shape
    #
    #     # set model to evaluation mode
    #     model.eval()
    #
    #     # summary for current eval loop
    #     # summ = []
    #     metrics = {}
    #
    #     # Load the checkpoint
    #     # self.model.load_w(self.checkpoint)
    #     # self.ckpt = torch.load(self.checkpoint)
    #     # self.model.load_state_dict(self.ckpt['net_dict'])
    #     # self.R = self.ckpt['R']
    #     # self.c = self.ckpt['c']
    #
    #     # Prepare a table to show results
    #     vad_table = self.empty_table
    #
    #     # Set up container for novelty scores from all test videos
    #     global_llk = []
    #     global_rec = []
    #     global_ns = []
    #     global_y = []
    #
    #     # Get accumulators，干嘛的？答：get frame-level scores from clip-level scores
    #     results_accumulator_llk = ResultsAccumulator(time_steps=t)
    #     results_accumulator_rec = ResultsAccumulator(time_steps=t)
    #
    #     cnt_step = 0  # global_step
    #     with SummaryWriter(log_dir="summary/test_{0}".format(
    #             params.output_file.split('.')[0]),
    #             comment="{}".format(params.dataset_name)) as writer:
    #         # Start iteration over test videos
    #         for cl_idx, video_id in enumerate(val_dataloader.test_videos):
    #             # test_videos 的内容是：TestXXX(XXX：001~012) 这些目录名，每个目录名保存有一个
    #             # 视频的所有帧，所以代表一个视频，即 video_id
    #
    #             # Run the test
    #             val_dataloader.test(video_id)  # 设置好cur_video_frames【其实是整个视频的全部clips】，
    #             # cur_video_gt，cur_len【其实是clips number】
    #             loader = DataLoader(val_dataloader,
    #                                 num_workers=1,
    #                                 shuffle=False,
    #                                 batch_size=1,
    #                                 collate_fn=val_dataloader.collate_fn)  # 临时构建loader
    #             # 因为是 inference，所以没有 batch_size (或者说==1)
    #             # collate_fn：实际作用是：TODO
    #
    #             # Build score containers
    #             sample_llk = np.zeros(shape=(len(loader) + t - 1,))
    #             sample_rec = np.zeros(shape=(len(loader) + t - 1,))
    #             # print("len(loader): ", len(loader)) # len(self.batch_sampler)
    #             # 因为loader会把所有Dataset的所有item都做登记，而len(dataset) ==
    #             # num_frames - t + 1，即所有的clips (带overlap的)，要恢复就是：
    #             # len(loader) + t - 1
    #             # print("len(loader) + t - 1: ", len(loader) + t - 1)
    #             sample_y = val_dataloader.load_test_sequence_gt(video_id)  # (n_frames,)
    #             # print("len(sample_y): ", len(sample_y))
    #             # 事实证明：(len(loader) + t - 1) == len(sample_y), len(loader) =
    #             #
    #             for i, (x, y) in tqdm(enumerate(loader),
    #                                   desc=f'Computing scores for {self.dataset_name}'):
    #                 #
    #                 cnt_step = cnt_step + 1
    #
    #                 x = x.to(self.device)
    #
    #                 x_r, z = model(x)
    #                 z = z.view(-1, 690, 128)
    #                 # print("in 327 line, z.size: ", z.size())
    #
    #                 ttloss = loss_fn(x, x_r, z)  # 记住，self.loss其实一个 object，这里
    #                 # 被执行了 forwrd()，所以等于修改了 object (即 self.loss被修改了)
    #                 total_loss = loss_fn.total_loss
    #                 reconstruction_loss = loss_fn.reconstruction_loss
    #                 deepSVDD_loss = loss_fn.deepSVDD_loss
    #                 # write all loss
    #                 # if cnt_step % Config.plot_every == 0:
    #                 #     writer.add_scalars("test_loss",
    #                 #                        {'total_loss': total_loss,
    #                 #                         'reconstruction_loss': reconstruction_loss,
    #                 #                         'autoregression_loss': autoregression_loss
    #                 #                         },
    #                 #                        cnt_step)
    #
    #                 # Feed results accumulators: 模仿一个队列，队尾进，队头出
    #                 # 我的办法：通过设置断点，进去看results_accumulator_llk是怎么工作的？
    #                 # 因为 batch_szie == 1, 所以push了 it(==num_clips==num_frames-t+1)次，
    #                 # 所以还有 (t - 1) 帧没有计算 loss，留到 下面的 while
    #                 results_accumulator_llk.push(loss_fn.deepSVDD_loss)
    #                 results_accumulator_rec.push(loss_fn.reconstruction_loss)
    #                 sample_llk[i] = results_accumulator_llk.get_next()
    #                 sample_rec[i] = results_accumulator_rec.get_next()
    #
    #             # Get last results
    #             # 计算剩下的 (t-1)帧各自的 loss
    #             while results_accumulator_llk.results_left != 0:
    #                 index = (- results_accumulator_llk.results_left)
    #                 sample_llk[index] = results_accumulator_llk.get_next()
    #                 sample_rec[index] = results_accumulator_rec.get_next()
    #
    #             min_llk, max_llk, min_rec, max_rec = self.compute_normalizing_coefficients(
    #                 sample_llk, sample_rec)
    #
    #             # Compute the normalized scores and novelty score
    #             sample_llk = normalize(sample_llk, min_llk, max_llk)
    #             sample_rec = normalize(sample_rec, min_rec, max_rec)
    #             sample_ns = novelty_score(sample_llk, sample_rec)
    #             # # 绘制 score-map
    #             # # print("len of sample_ns:", len(sample_ns))
    #             # fig_novelty_score = plt.figure()
    #             # plt.title('novelty_score of {}'.format(video_id))
    #             # plt.plot(range(len(sample_ns)), sample_ns, color='green',
    #             #          label='novelty_score')
    #             # plt.xlabel('frames')
    #             # plt.ylabel('novelty_score')
    #             # writer.add_figure('Novelty Score', fig_novelty_score, global_step=cl_idx)
    #
    #             # Update global scores (used for global metrics)
    #             global_llk.append(sample_llk)
    #             global_rec.append(sample_rec)
    #             global_ns.append(sample_ns)
    #             global_y.append(sample_y)
    #
    #             try:
    #                 # Compute AUROC for this video
    #                 this_video_metrics = [
    #                     roc_auc_score(sample_y, sample_llk),  # likelihood metric
    #                     roc_auc_score(sample_y, sample_rec),  # reconstruction metric
    #                     roc_auc_score(sample_y, sample_ns)  # novelty score
    #                 ]
    #                 vad_table.add_row([video_id] + this_video_metrics)
    #             except ValueError:
    #                 # This happens for sequences in which all frames are abnormal
    #                 # Skipping this row in the table (the sequence will still count for global metrics)
    #                 continue
    #
    #         # Compute global AUROC and print table
    #         global_llk = np.concatenate(global_llk)
    #         global_rec = np.concatenate(global_rec)
    #         global_ns = np.concatenate(global_ns)
    #         global_y = np.concatenate(global_y)
    #         global_metrics = [
    #             roc_auc_score(global_y, global_llk),  # likelihood metric
    #             roc_auc_score(global_y, global_rec),  # reconstruction metric
    #             roc_auc_score(global_y, global_ns)  # novelty score
    #         ]
    #         vad_table.add_row(['avg'] + list(global_metrics))
    #         print(vad_table)
    #
    #         # # Save table
    #         # with open(self.output_file, mode='w') as f:
    #         #     f.write(str(vad_table))
    #         #     #
    #         #     # 查看下网络
    #         #     # model_input = torch.rand([1380, 1, 8, 32, 32])
    #         #     # writer.add_graph(self.model, input_to_model=model_input)
    #         # print("ag_auc: ", list(global_metrics)[2])
    #         logging.info("avg_auc: ", list(global_metrics)[2])
    #         metrics['auc'] = list(global_metrics)[2]
    #         return  metrics# 返回 avg_auc