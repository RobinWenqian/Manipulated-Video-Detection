"""
Pytorch Dataloader for video. Trimmed over time and resizing the frames.
"""

from __future__ import print_function, division

import os
import pickle
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
from torchvision import transforms


class RandomCrop(object):
    """Randomly Crop the frames in a clip."""

    def __init__(self, output_size):
        """
            Args:
              output_size (tuple or int): Desired output size. If int, square crop
              is made.
        """
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, clip):
        h, w = clip.size()[2:]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        clip = clip[:, :, top : top + new_h, left : left + new_w]

        return clip


class GeneralVideoDataset(Dataset):
    """Dataset Class for Loading Video"""

    def __init__(
        self,
        clips_list_file,
        root_dir,
        channels,
        time_depth,
        mean, #(a,b,c) eg:(0.485, 0.456, 0.406)
        std, #(a,b,c) eg:(0.229, 0.224, 0.225)
        cut_size,
        transformflag,
        transform
    ):
        """
        Args:
            clips_list_file (string): Path to the clipsList file with labels.
            root_dir (string): Directory with all the videoes.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            channels: Number of channels of frames
            time_depth: Number of frames to be loaded in a sample
            x_size, y_size: Dimensions of the frames
            mean: Mean value of the training set videos over each channel
            std: Diviation of each channel
        """
        with open(clips_list_file, "rb") as fp:  # Unpickling
            clips_list_file = pickle.load(fp)

        self.clips_list = clips_list_file
        self.root_dir = root_dir
        self.channels = channels
        self.time_depth = time_depth
        self.mean = mean
        self.std = std
        self.cut_size = cut_size
        self.transformflag = transformflag
        self.transform = transform

    def __len__(self):
        return len(self.clips_list)

    def read_video(self, video_file):
        # Open the video file
        cap = cv2.VideoCapture(video_file)
        suc, frame = cap.read()
        #print(frame.shape)
        frames = torch.FloatTensor(
            self.channels, self.time_depth, frame.shape[0], frame.shape[1]
        )
        failed_clip = False
        for f in range(self.time_depth):
            suc, frame = cap.read()
            if suc:
                if self.transformflag:
                    frame = self.transform(frame)
                    frames[:, f, :, :] = frame
                else:
                    frame = torch.from_numpy(frame)
                    # HWC2CHW
                    frame = frame.permute(2, 0, 1)
                    frames[:, f, :, :] = frame
            else:
                print("Skipped!")
                failed_clip = True
                break

        return frames, failed_clip

    def __getitem__(self, idx):
        # Shuffle the video list
        self.clips_list = random.sample(self.clips_list,len(self.clips_list))

        video_file = os.path.join(self.root_dir, self.clips_list[idx][0])
        #print(self.clips_list[idx])
        clip, failed_clip = self.read_video(video_file)
        crop = RandomCrop(self.cut_size)
        clip_cut = crop(clip)
        sample = {
            "clip": clip_cut,
            "label": self.clips_list[idx][1],
            "failedClip": failed_clip,
        }

        clip_tuple = (sample['clip'],sample['label'])

        return clip_tuple

if __name__ == "__main__":
    clips_list_file = '../Celeb-DF-v2/List_of_testing_videos.pkl'
    root_dir = '../Celeb-DF-v2'
    channels = 3
    time_depth = 50
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transformflag = True
    transform = transforms.Compose([
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    transforms.Normalize(mean, std),
    ])

    cut_size = 225

    DataloaderTest = GeneralVideoDataset(
        clips_list_file,
        root_dir,
        channels,
        time_depth,
        mean,
        std,
        transformflag,
        transform)
    
    crop = RandomCrop(cut_size)
    for item in DataloaderTest:
        cut_clip = crop(item['clip'])
        print(cut_clip.shape)
