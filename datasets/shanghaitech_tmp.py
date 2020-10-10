import itertools
import os
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import exists
from os.path import getsize
from os.path import join

import cv2
import numpy as np
import skimage.io as io
from skimage.transform import resize
from torchvision import transforms
from tqdm import tqdm

from conf import Conf
from dataset.dataset_base import DatasetBase
from dataset.transforms import RemoveBackground
from dataset.transforms import RemoveBackgroundAndConcatMaskToY
from dataset.transforms import ToFloatTensor3D
from dataset.transforms import ToFloatTensor3DMask
from utils.generic import mkdir


class ShangaiVideo:

    def __init__(self, root, video_id):
        self.video_id = video_id
        self.rgb_path = join(root, 'frames', video_id)
        self.len = len(glob(join(self.rgb_path, '*.jpg')))


class SHANGHAITECH(DatasetBase):
    # fake epoch lengths
    n_train_examples = 1000
    n_val_examples = 100

    def __init__(self, cfg: Conf, which_one: str):

        super(SHANGHAITECH, self).__init__()

        assert which_one in ['rgb', 'rgb-fg', 'rgb-mask', 'rgb-masked-loss']

        self.path = join(cfg.majinbu_path, 'SHANGHAITECH')
        self.train_dir = join(self.path, 'training')
        self.test_dir = join(self.path, 'testing')
        self.which_one = which_one

        self._shapes = {
            'rgb': (3, 16, 256, 512),
            'rgb-fg': (3, 16, 256, 512),
            'rgb-mask': (4, 16, 256, 512),
            'rgb-masked-loss': (3, 16, 256, 512)
        }

        # record paths
        self.train_rgb_record_path = join(self.path, 'train_rgb.bin')
        self.train_mask_record_path = join(self.path, 'train_mask.bin')
        self.val_rgb_record_path = join(self.path, 'val_rgb.bin')
        self.val_mask_record_path = join(self.path, 'val_mask.bin')

        # load partitions
        self.train_ids = np.loadtxt(join(self.train_dir, 'train.txt'), dtype=str)
        self.val_ids = np.loadtxt(join(self.train_dir, 'val.txt'), dtype=str)
        self.test_ids = np.loadtxt(join(self.test_dir, 'test.txt'), dtype=str)

        self.train_videos = [
            ShangaiVideo(self.train_dir, video_id) for video_id in self.train_ids
        ]

        self.val_videos = [
            ShangaiVideo(self.train_dir, video_id) for video_id in self.val_ids
        ]

        # Directory for backgrounds
        self.train_background_dir = join(self.train_dir, 'backgrounds')
        self.test_background_dir = join(self.test_dir, 'backgrounds')

        # Actual backgrounds
        self.train_backgrounds = self.get_backgrounds('train')
        self.val_backgrounds = self.get_backgrounds('val')
        self.test_backgrounds = self.get_backgrounds('test')

        # Transform
        self.train_transform = None
        self.val_transform = None
        self.test_transform = None
        self._init_transform()

        # initialize currents
        self.cur_record_rgb = None
        self.cur_record_mask = None
        self.cur_len = 0
        self.cur_mode = None
        self.offset_probs = None
        self.transform = None
        self.test_video_id = None
        self.sampling_ids = None

    def _init_transform(self):
        if self.which_one == 'rgb':
            self.train_transform = transforms.Compose([
                RemoveBackground(threshold=128),
                ToFloatTensor3D(normalize=True)]
            )
            self.val_transform = transforms.Compose([
                RemoveBackground(threshold=128),
                ToFloatTensor3D(normalize=True)
            ])
            self.test_transform = transforms.Compose([
                RemoveBackground(threshold=128),
                ToFloatTensor3D(normalize=True)
            ])
        elif self.which_one == 'rgb-fg':
            self.train_transform = transforms.Compose([
                ToFloatTensor3D(normalize=True)]
            )
            self.val_transform = transforms.Compose([
                ToFloatTensor3D(normalize=True)
            ])
            self.test_transform = transforms.Compose([
                ToFloatTensor3D(normalize=True)
            ])
        elif self.which_one == 'rgb-mask':
            self.train_transform = transforms.Compose([
                ToFloatTensor3DMask(normalize=True)]
            )
            self.val_transform = transforms.Compose([
                ToFloatTensor3DMask(normalize=True)
            ])
            self.test_transform = transforms.Compose([
                ToFloatTensor3DMask(normalize=True)
            ])
        elif self.which_one == 'rgb-masked-loss':
            self.train_transform = transforms.Compose([
                RemoveBackgroundAndConcatMaskToY(threshold=128),
                ToFloatTensor3DMask(normalize=True, has_x_mask=False)
            ])
            self.val_transform = transforms.Compose([
                RemoveBackgroundAndConcatMaskToY(threshold=128),
                ToFloatTensor3DMask(normalize=True, has_x_mask=False)
            ])
            self.test_transform = transforms.Compose([
                RemoveBackgroundAndConcatMaskToY(threshold=128),
                ToFloatTensor3DMask(normalize=True, has_x_mask=False)
            ])
        else:
            raise ValueError

    def train(self):
        """ Sets the dataset in train mode. """
        self.cur_len = SHANGHAITECH.n_train_examples
        self.cur_mode = 'train'

        for record in [self.cur_record_rgb, self.cur_record_mask]:
            if record is not None:
                record.close()

        self.cur_record_rgb = open(self.train_rgb_record_path, mode='rb')
        self.cur_record_mask = open(self.train_mask_record_path, mode='rb')

        # get offset probs
        c, t, h, w = self.shape
        sampling_points = list(itertools.chain(*[
            [x for x in itertools.product([vid.len], np.arange(0, vid.len))]
            for vid in self.train_videos]))
        sampling_probs = np.array([int(v_len - t > f_idx) for v_len, f_idx in sampling_points], dtype=np.float64)

        sampling_ids = list(itertools.chain(*[
            [vid.video_id for _ in itertools.product([vid.len], np.arange(0, vid.len))]
            for vid in self.train_videos]))

        self.offset_probs = sampling_probs / np.sum(sampling_probs)
        self.transform = self.train_transform
        self.sampling_ids = sampling_ids

    def val(self):
        """ Sets the dataset in validation mode. """
        self.cur_len = SHANGHAITECH.n_val_examples
        self.cur_mode = 'val'

        for record in [self.cur_record_rgb, self.cur_record_mask]:
            if record is not None:
                record.close()

        self.cur_record_rgb = open(self.val_rgb_record_path, mode='rb')
        self.cur_record_mask = open(self.val_mask_record_path, mode='rb')

        # get offset probs
        c, t, h, w = self.shape
        sampling_points = list(itertools.chain(*[
            [x for x in itertools.product([vid.len], np.arange(0, vid.len))]
            for vid in self.val_videos]))
        sampling_probs = np.array([int(v_len - t > f_idx) for v_len, f_idx in sampling_points], dtype=np.float64)

        sampling_ids = list(itertools.chain(*[
            [vid.video_id for _ in itertools.product([vid.len], np.arange(0, vid.len))]
            for vid in self.val_videos]))

        self.offset_probs = sampling_probs / np.sum(sampling_probs)
        self.transform = self.val_transform
        self.sampling_ids = sampling_ids

    def test(self, video_id):
        """ Sets the dataset in test mode. """

        record_path_rgb = join(self.test_dir, 'records', '{}_rgb.bin'.format(video_id))
        record_path_mask = join(self.test_dir, 'records', '{}_mask.bin'.format(video_id))

        cc, t, w, h = self._shapes['rgb']
        # need to recover the length from the file dimension :(
        record_size_rgb = getsize(record_path_rgb)

        video_length_rgb = record_size_rgb / cc / h / w

        self.cur_len = int(video_length_rgb - t + 1)

        self.cur_mode = 'test'

        for record in [self.cur_record_rgb, self.cur_record_mask]:
            if record is not None:
                record.close()

        self.cur_record_rgb = open(record_path_rgb, mode='rb')
        self.cur_record_mask = open(record_path_mask, mode='rb')

        self.transform = self.test_transform
        self.test_video_id = video_id

    def gt_for(self, video_id: str) -> np.array:
        return np.load(join(self.test_dir, 'test_frame_mask', f'{video_id}.npy'))

    @property
    def shape(self):
        return self._shapes[self.which_one]

    def __len__(self):
        return self.cur_len

    def create_binary_records_rgb(self):

        c, t, h, w = self.shape

        train_vars = [self.train_rgb_record_path, self.train_videos, 'train']
        val_vars = [self.val_rgb_record_path, self.val_videos, 'val']

        for record_path, videos_list, name in [train_vars, val_vars]:
            if not exists(dirname(record_path)):
                os.makedirs(dirname(record_path))

            record = open(record_path, mode='wb+')

            for video in tqdm(videos_list, desc='Creating SHANGAITECH {} rgb record...'.format(name)):
                for image_path in sorted(glob(join(video.rgb_path, '*.jpg'))):
                    img = io.imread(fname=image_path)
                    img = resize(img, output_shape=(h, w), preserve_range=True)
                    img = np.uint8(img)
                    img.tofile(record)

            record.close()

    def create_binary_records_mask(self):

        c, t, h, w = self.shape

        train_vars = [self.train_mask_record_path, self.train_videos, 'train']
        val_vars = [self.val_mask_record_path, self.val_videos, 'val']

        for record_path, videos_list, name in [train_vars, val_vars]:
            if not exists(dirname(record_path)):
                os.makedirs(dirname(record_path))

            record = open(record_path, mode='wb+')

            for video in tqdm(videos_list, desc='Creating SHANGAITECH {} mask record...'.format(name)):
                img_list = sorted(glob(join(video.rgb_path, '*.jpg')))

                # Compute background
                mog = cv2.createBackgroundSubtractorMOG2(varThreshold=12)
                for frame in img_list:
                    img = cv2.resize(cv2.imread(frame), (w, h))
                    mog.apply(img)
                for i, frame in enumerate(img_list):
                    img = cv2.resize(cv2.imread(frame), (w, h))
                    fg_mask = mog.apply(img)
                    fg_mask[fg_mask == 127] = 0  # remove shadows
                    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
                    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                    fg_mask = fg_mask // 255  # -> {0, 1}
                    fg_mask.tofile(record)

            record.close()

    def create_test_records_rgb(self):

        c, t, h, w = self.shape
        output_dir = join(self.test_dir, 'records')
        mkdir(output_dir)
        for test_video in tqdm(glob(join(self.test_dir, 'frames', '*')),
                               desc='Creating SHANGAITECH test rgb records...'):
            # open record
            record = open(join(output_dir, '{}_rgb.bin'.format(basename(test_video))), mode='wb+')

            # image list
            image_list = sorted(glob(join(test_video, '*.jpg')))

            for image_path in image_list:
                img = io.imread(fname=image_path)
                img = resize(img, output_shape=(h, w), preserve_range=True)
                img = np.uint8(img)
                img.tofile(record)

            record.close()

    def create_test_records_mask(self):

        c, t, h, w = self.shape
        output_dir = join(self.test_dir, 'records')
        mkdir(output_dir)

        for test_video in tqdm(glob(join(self.test_dir, 'frames', '*')),
                               desc='Creating SHANGAITECH test mask records...'):
            # open record
            record = open(join(output_dir, '{}_mask.bin'.format(basename(test_video))), mode='wb+')

            # image list
            img_list = sorted(glob(join(test_video, '*.jpg')))

            # Compute background
            mog = cv2.createBackgroundSubtractorMOG2(varThreshold=12)
            for frame in img_list:
                img = cv2.resize(cv2.imread(frame), (w, h))
                mog.apply(img)
            for i, frame in enumerate(img_list):
                img = cv2.resize(cv2.imread(frame), (w, h))
                fg_mask = mog.apply(img)
                fg_mask[fg_mask == 127] = 0  # remove shadows
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                fg_mask = fg_mask // 255  # -> {0, 1}
                fg_mask.tofile(record)

            record.close()

    def get_backgrounds(self, partition: str):
        """ Returns dictionary like video_id: background """
        if partition == 'train':
            backgrounds_dict = {i: io.imread(join(self.train_background_dir, f'{i}.png'))[np.newaxis]
                                for i in self.train_ids}
        elif partition == 'val':
            backgrounds_dict = {i: io.imread(join(self.train_background_dir, f'{i}.png'))[np.newaxis]
                                for i in self.val_ids}
        elif partition == 'test':
            backgrounds_dict = {i: io.imread(join(self.test_background_dir, f'{i}.png'))[np.newaxis]
                                for i in self.test_ids}
        else:
            raise ValueError
        return backgrounds_dict

    def get_background_for(self, partition: str, video_id: str) -> np.array:
        if partition == 'train':
            background = self.train_backgrounds[video_id]
        elif partition == 'val':
            background = self.val_backgrounds[video_id]
        elif partition == 'test':
            background = self.test_backgrounds[video_id]
        else:
            raise ValueError
        return background

    def create_backgrounds(self):

        c, t, h, w = self.shape
        train_pars = (self.train_dir, self.train_background_dir, self.train_ids)
        val_pars = (self.train_dir, self.train_background_dir, self.val_ids)
        test_pars = (self.test_dir, self.test_background_dir, self.test_ids)
        for fr_dir, background_dir, ids in [train_pars, val_pars, test_pars]:
            for video_id in ids:
                video_path = join(fr_dir, 'frames', video_id)

                mog = cv2.createBackgroundSubtractorMOG2()
                for frame in sorted(glob(join(video_path, '*.jpg'))):
                    img = cv2.resize(cv2.imread(frame), (w, h))
                    mog.apply(img)

                # Get background
                background = mog.getBackgroundImage()

                # Save
                mkdir(background_dir)
                cv2.imwrite(join(background_dir, f'{video_id}.png'), background)

    def __getitem__(self, i):

        cc, t, h, w = self._shapes['rgb']

        while True:
            try:

                load_mask = (self.which_one in ['rgb-fg', 'rgb-mask'])

                if self.cur_mode == 'test':
                    index = i
                    background = self.test_backgrounds[self.test_video_id]
                elif self.cur_mode == 'train':
                    index = np.random.choice(np.arange(0, len(self.offset_probs)), p=self.offset_probs)
                    background = self.train_backgrounds[self.sampling_ids[index]]
                elif self.cur_mode == 'val':
                    index = np.random.choice(np.arange(0, len(self.offset_probs)), p=self.offset_probs)
                    background = self.val_backgrounds[self.sampling_ids[index]]

                else:
                    raise ValueError

                offset = index * (cc * h * w)

                self.cur_record_rgb.seek(offset)
                clip_rgb = np.fromfile(self.cur_record_rgb, dtype=np.uint8, count=(t * h * w * cc))
                clip_rgb = np.reshape(clip_rgb, newshape=(t, h, w, cc))

                # If the mask is needed load it and modify clip_rgb
                if load_mask:
                    offset = index * (h * w)  # no channels
                    self.cur_record_mask.seek(offset)
                    clip_mask = np.fromfile(self.cur_record_mask, dtype=np.uint8, count=(t * h * w))
                    clip_mask = np.reshape(clip_mask, newshape=(t, h, w, 1))

                    if self.which_one == 'rgb-fg':
                        clip_rgb *= clip_mask  # keep only foreground region
                    if self.which_one == 'rgb-mask':
                        clip_rgb *= clip_mask  # keep only
                        clip_rgb = np.concatenate((clip_rgb, clip_mask), axis=-1)

                # If has background clips, provide video_id to transform
                if self.has_remove_background_transform:
                    sample = clip_rgb, clip_rgb, background
                elif self.has_remove_background_and_concat_mask_to_y_transform:
                    sample = clip_rgb, clip_rgb, background
                else:
                    sample = clip_rgb, clip_rgb.copy()

                # Apply transform
                if self.transform:
                    sample = self.transform(sample)

                break
            except:
                continue

        return sample

    @property
    def log_frequency(self):
        return 500

    @property
    def max_log_size(self):
        return 6

    @property
    def validation_batches(self):
        return 16

    @property
    def task(self):
        return 'video-anomaly-detection'