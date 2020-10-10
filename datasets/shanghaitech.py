from glob import glob
from os.path import basename
from os.path import isdir
from os.path import join
from typing import List
from typing import Tuple

import cv2
import numpy as np
import skimage.io as io
import torch
from skimage.transform import resize
from torch.utils.data.dataloader import default_collate
from torchvision import transforms

from datasets.base import VideoAnomalyDetectionDataset
from datasets.transforms import RemoveBackground
from datasets.transforms import ToFloatTensor3D

# for Testing
# class SHANGHAITECH(VideoAnomalyDetectionDataset):
#     pass
# for Testing
class SHANGHAITECH(VideoAnomalyDetectionDataset):
    """
    Models ShanghaiTech dataset for video anomaly detection.
    """
    def __init__(self, path):
        # type: (str) -> None
        """
        Class constructor.

        :param path: The folder in which ShanghaiTech is stored.
        """
        super(SHANGHAITECH, self).__init__()

        self.path = path

        # Test directory
        self.test_dir = join(path, 'testing')

        # Transform
        self.transform = transforms.Compose([RemoveBackground(threshold=128),
                                             ToFloatTensor3D(normalize=True)])

        # Load all test ids
        self.test_ids = self.load_test_ids()

        # Other utilities
        self.cur_len = 0
        self.cur_video_id = None
        self.cur_video_frames = None
        self.cur_video_gt = None
        self.cur_background = None

    def load_test_ids(self):
        # type: () -> List[str]
        """
        Loads the set of all test video ids.

        :return: The list of test ids.
        """
        return sorted([basename(d) for d in glob(join(self.test_dir, 'frames', '**')) if isdir(d)])

    def load_test_sequence_frames(self, video_id):
        # type: (str) -> np.ndarray
        """
        Loads a test video in memory.

        :param video_id: the id of the test video to be loaded
        :return: the video in a np.ndarray, with shape (n_frames, h, w, c).
        """
        c, t, h, w = self.shape

        sequence_dir = join(self.test_dir,  'frames', video_id)
        img_list = sorted(glob(join(sequence_dir, '*.jpg')))
        test_clip = []
        for img_path in img_list:
            img = io.imread(img_path)
            img = resize(img, output_shape=(h, w), preserve_range=True)
            img = np.uint8(img)
            test_clip.append(img)
        test_clip = np.stack(test_clip)
        return test_clip

    def load_test_sequence_gt(self, video_id):
        # type: (str) -> np.ndarray
        """
        Loads the groundtruth of a test video in memory.

        :param video_id: the id of the test video for which the groundtruth has to be loaded.
        :return: the groundtruth of the video in a np.ndarray, with shape (n_frames,).
        """
        clip_gt = np.load(join(self.test_dir,  'test_frame_mask', f'{video_id}.npy'))
        return clip_gt

    def test(self, video_id):
        # type: (str) -> None
        """
        Sets the dataset in test mode.

        :param video_id: the id of the video to test.
        """
        c, t, h, w = self.shape

        self.cur_video_id = video_id
        self.cur_video_frames = self.load_test_sequence_frames(video_id)
        self.cur_video_gt = self.load_test_sequence_gt(video_id)
        self.cur_background = self.create_background(self.cur_video_frames)

        self.cur_len = len(self.cur_video_frames) - t + 1

    @property
    def shape(self):
        # type: () -> Tuple[int, int, int, int]
        """
        Returns the shape of examples being fed to the model.
        """
        return 3, 16, 256, 512

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
        return self.cur_len

    @staticmethod
    def create_background(video_frames):
        # type: (np.ndarray) -> np.ndarray
        """
        Create the background of a video via MOGs.

        :param video_frames: list of ordered frames (i.e., a video).
        :return: the estimated background of the video.
        """
        mog = cv2.createBackgroundSubtractorMOG2()
        for frame in video_frames:
            img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            mog.apply(img)

        # Get background
        background = mog.getBackgroundImage()

        return cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

    def __getitem__(self, i):
        # type: (int) -> Tuple[torch.Tensor, torch.Tensor]
        """
        Provides the i-th example.
        """
        c, t, h, w = self.shape

        clip = self.cur_video_frames[i:i+t]

        sample = clip, clip, self.cur_background

        # Apply transform
        if self.transform:
            sample = self.transform(sample)

        return sample

    @property
    def collate_fn(self):
        """
        Returns a function that decides how to merge a list of examples in a batch.
        """
        return default_collate

    def __repr__(self):
        return f'ShanghaiTech (video id = {self.cur_video_id})'


#########################################################################

# for Training by HaoZhang
class SHANGHAITECH_TRAIN(VideoAnomalyDetectionDataset):
    """
    Models ShanghaiTech dataset for video anomaly detection.
    """
    def __init__(self, path):
        # type: (str) -> None
        """
        Class constructor.

        :param path: The folder in which ShanghaiTech is stored.
        """
        super(SHANGHAITECH_TRAIN, self).__init__()

        self.path = path

        # Test directory
        self.train_dir = join(path, 'training')

        # Transform
        self.transform = transforms.Compose([RemoveBackground(threshold=128),
                                             ToFloatTensor3D(normalize=True)])

        # Load all test ids
        self.train_ids = self.load_train_ids()

        # Other utilities
        self.cur_len = 0
        self.cur_video_id = None
        self.cur_video_frames = None
        # self.cur_video_gt = None
        self.cur_background = None

    def load_train_ids(self):
        # type: () -> List[str]
        """
        Loads the set of all test video ids.

        :return: The list of test ids.
        """
        return sorted([basename(d) for d in
                       glob(join(self.train_dir, 'frames', '**')) if isdir(d)])

    def load_train_sequence_frames(self, video_id):
        # type: (str) -> np.ndarray
        """
        Loads a test video in memory.

        :param video_id: the id of the test video to be loaded
        :return: the video in a np.ndarray, with shape (n_frames, h, w, c).
        """
        c, t, h, w = self.shape

        sequence_dir = join(self.train_dir,  'frames', video_id)
        img_list = sorted(glob(join(sequence_dir, '*.jpg')))
        train_clip = []
        for img_path in img_list:
            img = io.imread(img_path)
            img = resize(img, output_shape=(h, w), preserve_range=True)
            img = np.uint8(img)
            train_clip.append(img)
        train_clip = np.stack(train_clip)
        return train_clip

    # def load_test_sequence_gt(self, video_id):
    #     # type: (str) -> np.ndarray
    #     """
    #     Loads the groundtruth of a test video in memory.
    #
    #     :param video_id: the id of the test video for which the groundtruth has to be loaded.
    #     :return: the groundtruth of the video in a np.ndarray, with shape (n_frames,).
    #     """
    #     clip_gt = np.load(join(self.train_dir,  'test_frame_mask', f'{video_id}.npy'))
    #     return clip_gt

    def train(self, video_id):
        # type: (str) -> None
        """
        Sets the dataset in test mode.

        :param video_id: the id of the video to test.
        """
        c, t, h, w = self.shape

        self.cur_video_id = video_id
        self.cur_video_frames = self.load_train_sequence_frames(video_id)
        # self.cur_video_gt = self.load_test_sequence_gt(video_id)
        self.cur_background = self.create_background(self.cur_video_frames)
        self.cur_len = len(self.cur_video_frames) - t + 1

    @property
    def shape(self):
        # type: () -> Tuple[int, int, int, int]
        """
        Returns the shape of examples being fed to the model.
        """
        return 3, 16, 256, 512

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
        return self.cur_len

    @staticmethod
    def create_background(video_frames):
        # type: (np.ndarray) -> np.ndarray
        """
        Create the background of a video via MOGs.

        :param video_frames: list of ordered frames (i.e., a video).
        :return: the estimated background of the video.
        """
        mog = cv2.createBackgroundSubtractorMOG2()
        for frame in video_frames:
            img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            mog.apply(img)

        # Get background
        background = mog.getBackgroundImage()

        return cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

    def __getitem__(self, i):
        # type: (int) -> Tuple[torch.Tensor, torch.Tensor]
        """
        Provides the i-th example.
        """
        c, t, h, w = self.shape

        clip = self.cur_video_frames[i:i+t]

        sample = clip, clip, self.cur_background

        # Apply transform
        if self.transform:
            sample = self.transform(sample) # 会自动减去cur_background，返回(X,Y)

        return sample

    @property
    def collate_fn(self):
        """
        Returns a function that decides how to merge a list of examples in a batch.
        """
        return default_collate

    def __repr__(self):
        return f'ShanghaiTech (video id = {self.cur_video_id})'


# 测试小程序，By HaoZhang
#####################################################################################
def test_TRAIN():
    dataset = SHANGHAITECH_TRAIN(path='/home/zh/Papers_Code/CVPR2019_pytorch_VAD/'
                                      'novelty-detection/data/shanghaitech')
    print("self.train_dir: ", dataset.train_dir)
    ids = dataset.load_train_ids()
    print("ids: ", ids) # (01_001~13_007)
    #
    clip = dataset.load_train_sequence_frames("13_007")
    print("clip.shape: ", clip.shape) # (1105, 3, 256, 512) # t,c,h,w
    #
    dataset.train("13_007")
    print("Train001的cur_len", dataset.cur_len) #理想是1105-16+1 = 1090
    # 实际是：1090，逻辑正确
    #
    clip_0 = dataset[0]
    # print("clip_0的shape: ", clip_0[0].shape, clip_0[1].shape, clip_0[2].shape)
    print("clip_0的shape: ", clip_0[0].shape, clip_0[1].shape)
    # 没有加入self.transform，是这样的：(16, 256, 512, 3) (16, 256, 512, 3)
    #                                   (256, 512, 3)
    # 加入 self.transform，这样的：torch.Size([3, 16, 256, 512]) # [c,t,h,w]
    #                              torch.Size([3, 16, 256, 512])
    # TODO 本论文的transform看来有必要认真读下！！！
    #

def test_loader():
    from torch.utils.data import DataLoader
    dataset = SHANGHAITECH_TRAIN(path='/home/zh/Papers_Code/CVPR2019_pytorch_VAD/'
                                      'novelty-detection/data/shanghaitech')
    for cl_idx, video_id in enumerate(dataset.train_videos):
        dataset.train(video_id)
        loader = DataLoader(dataset,
                            collate_fn=dataset.collate_fn,
                            num_workers=4,
                            batch_size=8,
                            shuffle=True)
        # print("hhh")
        for i, x in enumerate(loader):
            # print("hhh2")
            # print("x.len",len(x))
            # print("x: ", x[0].shape, x[1].shape, x[2].shape)
            # torch.Size([8, 16, 256, 512, 3])
            # torch.Size([8, 16, 256, 512, 3])
            # torch.Size([8, 256, 512, 3]), 奇怪这个 background怎么也进来了？？？
            # 懂了，我把 transform注释掉了。。。刚才做测试的时候。。。
            #
            print("x: ", x[0].shape, x[1].shape)
            # print("x, y", x.shape, y.shape)

if __name__ == '__main__':
    # for Testing set
    # test_test_load_test_ids()
    ##########################################################
    # for Training
    # test_TRAIN()
    test_loader()

