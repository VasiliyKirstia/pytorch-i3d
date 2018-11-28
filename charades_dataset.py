import json
import os
import os.path
import math
import pandas as pd
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    
    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))


def load_rgb_frames(image_dir, vid, start, num):
    frames = []
    for i in range(start, start + num):
        img = cv2.imread(os.path.join(image_dir, vid, vid + '-' + str(i).zfill(6) + '.jpg'))[:, :, [2, 1, 0]]
        w, h, c = img.shape
        if w < 226 or h < 226:
            d = 226. - min(w, h)
            sc = 1 + d / min(w, h)
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
        img = (img / 255.) * 2 - 1
        frames.append(img)
    return np.asarray(frames, dtype=np.float32)


def load_flow_frames(image_dir, vid, start, num):
    frames = []
    for i in range(start, start + num):
        imgx = cv2.imread(os.path.join(image_dir, vid, vid + '-' + str(i).zfill(6) + 'x.jpg'), cv2.IMREAD_GRAYSCALE)
        imgy = cv2.imread(os.path.join(image_dir, vid, vid + '-' + str(i).zfill(6) + 'y.jpg'), cv2.IMREAD_GRAYSCALE)

        w, h = imgx.shape
        if w < 224 or h < 224:
            d = 224. - min(w, h)
            sc = 1 + d / min(w, h)
            imgx = cv2.resize(imgx, dsize=(0, 0), fx=sc, fy=sc)
            imgy = cv2.resize(imgy, dsize=(0, 0), fx=sc, fy=sc)

        imgx = (imgx / 255.) * 2 - 1
        imgy = (imgy / 255.) * 2 - 1
        img = np.asarray([imgx, imgy]).transpose([1, 2, 0])
        frames.append(img)
    return np.asarray(frames, dtype=np.float32)


def make_dataset(split_file, split, root, mode, num_classes=157):
    dataset = []
    with open(split_file, 'r') as f:
        data = json.load(f)

    i = 0
    for vid in data.keys():
        if data[vid]['subset'] != split:
            continue

        if not os.path.exists(os.path.join(root, vid)):
            continue
        num_frames = len(os.listdir(os.path.join(root, vid)))
        if mode == 'flow':
            num_frames = num_frames // 2

        if num_frames < 66:
            continue

        label = np.zeros((num_classes, num_frames), np.float32)

        fps = num_frames / data[vid]['duration']
        for ann in data[vid]['actions']:
            for fr in range(0, num_frames, 1):
                if fr / fps > ann[1] and fr / fps < ann[2]:
                    label[ann[0], fr] = 1  # binary classification
        dataset.append((vid, label, data[vid]['duration'], num_frames))
        i += 1

    return dataset


class Charades(Dataset):

    @staticmethod
    def preprocess_csv(df):
        df = df.join(df['actions'].str.split(';', expand=True))
        df.drop('actions', axis=1, inplace=True)
        df = df.melt(id_vars=['id'])
        df.drop('variable', axis=1, inplace=True)
        df.sort_values('id', inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.dropna(inplace=True)

        # df.columns == ['id', 'value']

        df = df.join(df['value'].str.split(' ', expand=True))
        df.drop('value', axis=1, inplace=True)
        df.rename({0: 'target', 1: 'start', 2: 'end'}, axis=1, inplace=True)
        df.reset_index(drop=True, inplace=True)
        df['start'] = df['start'].astype(np.float32)
        df['end'] = df['end'].astype(np.float32)

        return df  # df.columns == ['id', 'target', 'start', 'end']

    @staticmethod
    def target_to_integer(target_: str):
        return int(target_[1:], base=10)

    def __init__(
        self,
        data_csv_path,
        data_directory,
        transformation=None,
        video_ext='.mp4',
        random_start=True,
        frame_step=1,
        batch_size=64,
        random_seed=42,
    ):
        """
        Charades Dataset
        :param data_csv_path: path to Charades_v1_<train/test>.csv
        :param data_directory: path to Charades_v1_480 directory (directory with video files)
        :param transformation: transformation
        :param video_ext: extinsion of video files
        :param random_start: first frame of sequence will be choosen randomly
        :param frame_step: if bigger than 1, each frame_step-th frame will be returned
        :param batch_size: number of frames over temporal dimension
        :param random_seed: random seed
        """
        self.data = self.preprocess_csv(pd.read_csv(data_csv_path, usecols=['id', 'actions']))
        self.data_directory = data_directory if data_directory.endswith('/') else data_directory + '/'
        self.transformation = transformation
        self.video_ext = video_ext
        self.random_start = random_start
        self.frame_step = frame_step
        self.batch_size = batch_size
        self.random_state = np.random.RandomState(random_seed)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (video (batch of 64 frames, small videos will be repeated to acheave 64 frames length), target)
            where target is class_index of the target class.
        """
        video_id, target_, start, end = tuple(self.data.loc[index, ['id', 'target', 'start', 'end']])
        video_capture = cv2.VideoCapture(f'{self.data_directory}{video_id}{self.video_ext}')
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        frames_in_interval = math.floor((end - start) * fps)
        gap = max(0, frames_in_interval - self.batch_size)

        if self.random_start:
            start += self.random_state.randint(gap + 1)/fps

        frames = []
        video_capture.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
        frames_captured = 0
        while frames_captured < self.batch_size:
            if end * 1000 < video_capture.get(cv2.CAP_PROP_POS_MSEC):
                video_capture.set(cv2.CAP_PROP_POS_MSEC, start * 1000)

            _, frame = video_capture.read()

            if frame is not None:
                frames.append(frame)
                frames_captured += 1
            else:
                video_capture.set(cv2.CAP_PROP_POS_MSEC, start * 1000)

        frames = np.array(frames, dtype=np.float32)
        if self.transformation is not None:
            frames = self.transformation(frames)
        return frames, self.target_to_integer(target_)

    def __len__(self):
        return len(self.data.index)
