from glob import glob
from os.path import basename
from os.path import isdir
from os.path import join
from typing import List
from typing import Tuple

import numpy as np
import skimage.io as io
import torch
from skimage.transform import resize
from torchvision import transforms

from datasets.base import VideoAnomalyDetectionDataset
from datasets.transforms import ToCrops
from datasets.transforms import ToFloatTensor3D
from utils import concat_collate

# For Testing
# class UCSDPed2(VideoAnomalyDetectionDataset):
#     pass

class UCSDPed2(VideoAnomalyDetectionDataset):
    """
    Models UCSD Ped2 dataset for video anomaly detection.
    """
    def __init__(self, path):
        # type: (str) -> None
        """
        Class constructor.

        :param path: The folder in which UCSD is stored.
        """
        super(UCSDPed2, self).__init__()

        self.path = join(path, 'UCSDped2')

        # Test directory
        self.test_dir = join(self.path, 'Test')

        # Transform
        self.transform = transforms.Compose([ToFloatTensor3D(), ToCrops(self.raw_shape, self.crop_shape)])

        # Load all test ids
        self.test_ids = self.load_test_ids()

        # Other utilities
        self.cur_len = 0
        self.cur_video_id = None
        self.cur_video_frames = None
        self.cur_video_gt = None

    def load_test_ids(self):
        # type: () -> List[str]
        """
        Loads the set of all test video ids.

        ############################################################################################################
        # By HaoZhang
        # basename用于去掉目录的路径，只返回文件名: e.g. os.path.basename('d:\\library\\book.txt') => 'book.txt'
        # glob(join(self.test_dir, '**')： 返回路径下{self.test_dir/'**'}的路径名，
        *：匹配前一个表达式0次或多次。等价于 {0,}。
        # 结合下面的if，本函数的功能：返回self.test_dir下所有目录名，但是除了带gt的，再根据UCSD Dataset 结构特点反推
        # 这个是返回 UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test 下面的所有 TestXXX(XXX：001~012) 目录名
        ############################################################################################################
        :return: The list of test ids.
        """
        return sorted([basename(d) for d in glob(join(self.test_dir, '**'))
                       if isdir(d) and 'gt' not in basename(d)])

    def load_test_sequence_frames(self, video_id):
        # type: (str) -> np.ndarray
        """
        Loads a test video in memory.

        :param video_id: the id of the test video to be loaded
        :return: the video in a np.ndarray, with shape (n_frames, h, w, c).
        """
        c, t, h, w = self.raw_shape

        sequence_dir = join(self.test_dir, video_id)
        # By HaoZhang, for UCSD/pde1 or ped2, video_id: TestXXX(XXX：001~012)
        img_list = sorted(glob(join(sequence_dir, '*.tif'))) # 当前 vid下面所有tif
        test_clip = []
        for img_path in img_list: # 处理所有的 tif image ???
            img = io.imread(img_path)
            img = resize(img, output_shape=(h, w), preserve_range=True) # preserve_range : 是否保留原来的value范围
            img = np.uint8(img)
            test_clip.append(img) # list
        test_clip = np.stack(test_clip) # 合并为更高一维的tensor, 把所有的frame都合并到一个大 tensor中，这种处理很棒
        # 我之前的处理是，[start,end]，仅仅合并16帧为一个clip，逐个获取 test clip 【分治处理所有的帧】
        return test_clip  # 所有帧构成的 4-D tensor: (n_frames, h, w, c)，然后根据需要再提取部分 clip
        # {这个做法很好},但存在一个隐患和一个待处理逻辑：一个隐患是将当前子目录所有图片全部读入内存，会不会内存爆炸
        # 一个遗留问题：这个每次都要带上 video_id 才能获取到video_id的所有帧

    def load_test_sequence_gt(self, video_id):
        # type: (str) -> np.ndarray
        """
        Loads the groundtruth of a test video in memory.

        :param video_id: the id of the test video for which the groundtruth has to be loaded.
        :return: the groundtruth of the video in a np.ndarray, with shape (n_frames,).
        """
        sequence_dir = join(self.test_dir, f'{video_id}_gt')
        # By HaoZhang, for UCSD/pde1 or ped2, video_id: TestXXX(XXX：001~012)
        img_list = sorted(glob(join(sequence_dir, '*.bmp')))
        clip_gt = []
        for img_path in img_list:
            img = io.imread(img_path) // 255  # 5 // 2 == 2.5
            clip_gt.append(np.max(img))  # if at least one pixel is 1, then anomaly
        clip_gt = np.stack(clip_gt)
        return clip_gt

    def test(self, video_id):
        # type: (str) -> None
        """
        Sets the dataset in test mode.

        :param video_id: the id of the video to test.
        """
        c, t, h, w = self.raw_shape

        self.cur_video_id = video_id
        self.cur_video_frames = self.load_test_sequence_frames(video_id)
        self.cur_video_gt = self.load_test_sequence_gt(video_id)

        self.cur_len = len(self.cur_video_frames) - t + 1 # # 不是vid下面所有帧构成的clip ?? 怎么求 len(clips)?
        # 唯一答案： len(tensor) == tensor.size()[0] or tensor.shape[0]
        # 另外，这个 cur_len 的算法核心是，举例： [1,2,3,4,5]以3为window_size，得到clips依次为：
        # [1,2,3], [2,3,4], [3,4,5], 即 total_len - window_size + 1 (本例子中是：5 - 3 + 1 == 3)

    @property
    def shape(self):
        # type: () -> Tuple[int, int, int, int]
        """
        Returns the shape of examples being fed to the model.
        """
        return self.crop_shape

    @property
    def raw_shape(self):
        # type: () -> Tuple[int, int, int, int]
        """
        Returns the shape of the raw examples (prior to patches).
        """
        return 1, 16, 256, 384 # TODO：疑问，UCSD iamge 的原始大小是： 238x158x1, 那这里 raw_shape 是 ？？

    @property
    def crop_shape(self):
        # type: () -> Tuple[int, int, int, int]
        """
        Returns the shape of examples (patches).
        """
        return 1, 8, 32, 32 # TODO：这个 crop的物理过程到底是怎样？

    @property
    def test_videos(self):
        # type: () -> List[str]
        """
        Returns all available test videos.
        """
        return self.test_ids

    def __len__(self):
        # type: () -> int
        """
        Returns the number of examples.
        """
        return int(self.cur_len)

    def __getitem__(self, i):
        # type: (int) -> Tuple[torch.Tensor, torch.Tensor]
        """
        Provides the i-th example.
        """
        c, t, h, w = self.raw_shape

        clip = self.cur_video_frames[i:i+t] # 因为默认从第一个维度t 做切片slince
        clip = np.expand_dims(clip, axis=-1)  # add channel dimension, 为什么升维？？？TODO
        sample = clip, clip # 重构模型 x->x

        if self.transform:
            sample = self.transform(sample)

        return sample

    @property
    def collate_fn(self):
        """
        Returns a function that decides how to merge a list of examples in a batch.
        """
        return concat_collate

    def __repr__(self):
        return f'UCSD Ped2 (video id = {self.cur_video_id})'

# 测试的小代码段
def test_test_load_test_ids():
    dataset = UCSDPed2(path='/home/zh/Papers_Code/CVPR2019_pytorch_VAD'
                            '/novelty-detection/data/UCSD_Anomaly_Dataset.v1p2')
    print("self.test_ids: ", dataset.test_ids)
    res = dataset.load_test_ids()
    print("test_load_test_ids,res: ", res)

#####################################################################################

# For Training by HaoZhang
class UCSDPed2_TRAIN(VideoAnomalyDetectionDataset):
    """
    Models UCSD Ped2 dataset for video anomaly detection.
    """

    def __init__(self, path):
        # type: (str) -> None
        """
        Class constructor.

        :param path: The folder in which UCSD is stored.
        """
        super(UCSDPed2_TRAIN, self).__init__()

        self.path = join(path, 'UCSDped2') # 这个path通用于 Train or Test

        # Train directory
        self.train_dir = join(self.path, 'Train')

        # Transform
        # TODO: Training还需要其他的Data Augmentation 吗？ 问下论文作者！
        self.transform = transforms.Compose([ToFloatTensor3D(), ToCrops(self.raw_shape, self.crop_shape)])

        # Load all train ids
        self.train_ids = self.load_train_ids()

        # Other utilities
        self.cur_len = 0
        self.cur_video_id = None # Train 下面的所有 TrainXXX(XXX：001~016) 目录名
        self.cur_video_frames = None
        # self.cur_video_gt = None

    def load_train_ids(self):
        # type: () -> List[str]
        """
        Loads the set of all test video ids.

        ############################################################################################################
        # By HaoZhang
        # basename用于去掉目录的路径，只返回文件名: e.g. os.path.basename('d:\\library\\book.txt') => 'book.txt'
        # glob(join(self.test_dir, '**')： 返回路径下{self.test_dir/'**'}的路径名，
        *：匹配前一个表达式0次或多次。等价于 {0,}。
        # 结合下面的if，本函数的功能：返回self.test_dir下所有目录名，但是除了带gt的，再根据UCSD Dataset 结构特点反推
        # 这个是返回 UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train 下面的所有 TrainXXX(XXX：001~016) 目录名
        ############################################################################################################
        :return: The list of test ids.
        """
        return sorted([basename(d) for d in glob(join(self.train_dir, '**')) if isdir(d)])

    def load_train_sequence_frames(self, video_id):
        # type: (str) -> np.ndarray
        """
        Loads a test video in memory.

        :param video_id: the id of the test video to be loaded
        :return: the video in a np.ndarray, with shape (n_frames, h, w, c).
        """
        c, t, h, w = self.raw_shape

        sequence_dir = join(self.train_dir, video_id)
        # By HaoZhang, for UCSD/pde1 or ped2, video_id: TrainXXX(XXX：001~016)
        img_list = sorted(glob(join(sequence_dir, '*.tif')))  # 当前 vid下面所有tif
        train_clip = []
        for img_path in img_list:  # 处理所有的 tif image ??? 对的
            img = io.imread(img_path)
            img = resize(img, output_shape=(h, w), preserve_range=True)  # preserve_range : 是否保留原来的value范围
            img = np.uint8(img)
            train_clip.append(img)  # list
        train_clip = np.stack(train_clip)  # 合并为更高一维的tensor, 把所有的frame都合并到一个大 tensor中，这种处理很棒
        # 我之前的处理是，[start,end]，仅仅合并16帧为一个clip，逐个获取 test clip 【分治处理所有的帧】
        return train_clip  # 所有帧构成的 4-D tensor: (n_frames, h, w, c)，然后根据需要再提取部分 clip
        # {这个做法很好},但存在一个隐患和一个待处理逻辑：一个隐患是将当前子目录所有图片全部读入内存，会不会内存爆炸
        # 一个遗留问题：这个每次都要带上 video_id 才能获取到video_id的所有帧

    # def load_test_sequence_gt(self, video_id):
    #     # type: (str) -> np.ndarray
    #     """
    #     Loads the groundtruth of a test video in memory.
    #
    #     :param video_id: the id of the test video for which the groundtruth has to be loaded.
    #     :return: the groundtruth of the video in a np.ndarray, with shape (n_frames,).
    #     """
    #     sequence_dir = join(self.test_dir, f'{video_id}_gt')
    #     # By HaoZhang, for UCSD/pde1 or ped2, video_id: TestXXX(XXX：001~012)
    #     img_list = sorted(glob(join(sequence_dir, '*.bmp')))
    #     clip_gt = []
    #     for img_path in img_list:
    #         img = io.imread(img_path) // 255  # 5 // 2 == 2.5
    #         clip_gt.append(np.max(img))  # if at least one pixel is 1, then anomaly
    #     clip_gt = np.stack(clip_gt)
    #     return clip_gt

    def train(self, video_id):
        # type: (str) -> None
        """
        Sets the dataset in test mode.

        :param video_id: the id of the video to test.
        """
        c, t, h, w = self.raw_shape # 默认clip_len为16

        self.cur_video_id = video_id
        self.cur_video_frames = self.load_train_sequence_frames(video_id)
        # self.cur_video_gt = self.load_test_sequence_gt(video_id)

        self.cur_len = len(self.cur_video_frames) - t + 1  # # 不是vid下面所有帧构成的clip ?? 怎么求 len(clips)?
        # 唯一答案： len(tensor) == tensor.size()[0] or tensor.shape[0]
        # 经过代码测试，我的上述猜想正确
        # 另外，这个 cur_len 的算法核心是，举例： [1,2,3,4,5]以3为window_size，得到clips依次为：
        # [1,2,3], [2,3,4], [3,4,5], 即 total_len - window_size + 1 (本例子中是：5 - 3 + 1 == 3)

    @property
    def shape(self):
        # type: () -> Tuple[int, int, int, int]
        """
        Returns the shape of examples being fed to the model.
        """
        return self.crop_shape

    @property
    def raw_shape(self):
        # type: () -> Tuple[int, int, int, int]
        """
        Returns the shape of the raw examples (prior to patches).
        """
        return 1, 16, 256, 384  # TODO：疑问，UCSD iamge 的原始大小是： 238x158x1, 那这里 raw_shape 是 ？？

    @property
    def crop_shape(self):
        # type: () -> Tuple[int, int, int, int]
        """
        Returns the shape of examples (patches).
        """
        return 1, 8, 32, 32  # TODO：这个 crop的物理过程到底是怎样？8是什么意思？

    @property
    def train_videos(self):
        # type: () -> List[str]
        """
        Returns all available test videos.
        """
        return self.train_ids

    def __len__(self):
        # type: () -> int
        """
        Returns the number of examples.
        """
        return int(self.cur_len)

    def __getitem__(self, i):
        # type: (int) -> Tuple[torch.Tensor, torch.Tensor]
        """
        Provides the i-th example.
        """
        c, t, h, w = self.raw_shape

        clip = self.cur_video_frames[i:i + t]  # 因为默认从第一个维度t 做切片slince
        clip = np.expand_dims(clip, axis=-1)  # add channel dimension, 为什么升维？？？TODO
        sample = clip, clip  # 重构模型 x->x

        if self.transform:
            sample = self.transform(sample)

        return sample

    @property
    def collate_fn(self):
        """
        Returns a function that decides how to merge a list of examples in a batch.
        """
        return concat_collate

    def __repr__(self):
        return f'UCSD Ped2 (video id = {self.cur_video_id})'


# by HaoZhang
# 发现测试不了，应该是代码结构的问题
# 不是，其实是环境包依赖问题，花点时间逐个解决就OK了
def test_TRAIN():
    dataset = UCSDPed2_TRAIN(path='/home/zh/Papers_Code/CVPR2019_pytorch_VAD'
                                  '/novelty-detection/data/UCSD_Anomaly_Dataset.v1p2')
    print("self.train_dir: ", dataset.train_dir)
    ids = dataset.load_train_ids()
    print("ids: ", ids)  # Trainxxx (001~016)
    #
    clip = dataset.load_train_sequence_frames("Train001")
    print("clip.shape: ", clip.shape)  # (120, 256, 384)
    #
    dataset.train("Train001")
    print("Train001的cur_len", dataset.cur_len)  # 理想是120-16+1 = 105
    # 实际是：105，逻辑正确
    #
    clip_0 = dataset[0]
    print("clip_0的shape: ", clip_0[0].shape, clip_0[1].shape)
    # 没有加入self.transform，是这样的：(16, 256, 384, 1) (16, 256, 384, 1)
    # 加入 self.transform，这样的：torch.Size([690, 1, 8, 32, 32]) torch.Size([690, 1, 8, 32, 32])
    # TODO 本论文的transform看来有必要认真读下！！！
    #
####################################################################################


if __name__ == '__main__':
    ##########################################################
    # for Testing set
    # test_test_load_test_ids()
    ##########################################################
    # for Training
    test_TRAIN()
