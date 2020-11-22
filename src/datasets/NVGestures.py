import torch

from torch.utils.data.dataset import Dataset

import numpy as np
import cv2

from datasets.utils.read_data import load_split_nvgesture, load_data_from_file
from datasets.utils.normals import normals_multi
from datasets.utils.normalize import normalize

from pathlib import Path

class NVGesture(Dataset):
    """NVGesture Dataset class"""
    def __init__(self, configer, path, split="train", data_type="depth", transforms=None, n_frames=40, optical_flow=False):
        """Constructor method for NVGesture Dataset class

        Args:
            configer (Configer): Configer object for current procedure phase (train, test, val)
            split (str, optional): Current procedure phase (train, test, val)
            data_type (str, optional): Input data type (depth, rgb, normals, ir)
            transform (Object, optional): Data augmentation transformation for every data
            n_frames (int, optional): Number of frames selected for every input clip
            optical_flow (bool, optional): Flag to choose if calculate optical flow or not

        """
        super().__init__()

        print("Loading NVGestures {} dataset...".format(split.upper()), end=" ")

        self.dataset_path = Path(path) / "nvgesture_arch" / "nvGesture_v1"
        self.split = split
        self.data_type = data_type
        self.transforms = transforms
        self.optical_flow = optical_flow
        if self.data_type in ["normal", "normals"] and self.optical_flow:
            raise NotImplementedError("Optical flow for normals image is not supported.")

        file_lists = self.dataset_path / \
                     "nvgesture_{}_correct_cvpr2016_v2.lst".format(self.split if self.split == "train" else "test")

        self.data_list = list()
        load_split_nvgesture(file_with_split=str(file_lists), list_split=self.data_list)

        if self.data_type in ["depth_z", "depth", "normal", "normals"]:
            self.sensor = "depth"
        elif self.data_type == "wrapped":
            self.sensor = "wrapped"
        elif self.data_type in ["rgb", "color"]:
            self.sensor = "color"
        elif self.data_type == "ir":
            self.sensor = "duo_left"
        else:
            raise NotImplementedError
        print("done.")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data, label, offsets = load_data_from_file(self.dataset_path, example_config=self.data_list[idx], sensor=self.sensor,
                                          image_width=320, image_height=240)
        if self.optical_flow:
            if self.transforms:
                aug_det = self.transforms.to_deterministic()
                data = np.array([aug_det.augment_image(data[..., i])
                                 for i in range(data.shape[-1])]).transpose(1, 2, 3, 0)
            prev = data[..., 0]
            if self.data_type == "rgb":
                prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
            data = data[..., [0, 1] + [*range(2, data.shape[-1], 2)]]
            flow = np.zeros((data.shape[0], data.shape[1], 3, data.shape[-1] - 1))
            for i in range(1, data.shape[-1]):
                next = data[..., i]
                if self.data_type == "rgb":
                    next = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)
                flow[..., i - 1] = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                prev = next

        data = data[..., [*range(0, data.shape[-1], 2)]]  # Our settings is working with static clip containing 40 frames

        if self.data_type in ["normal", "normals"]:
            data = normals_multi(data)
        else:
            data = normalize(data)

        if self.transforms is not None and not self.optical_flow:
            aug_det = self.transforms.to_deterministic()
            data = np.array([aug_det.augment_image(data[..., i]) for i in range(data.shape[-1])]).transpose(1, 2, 3, 0)

        data = np.concatenate(data.transpose(3, 0, 1, 2), axis=2).transpose(2, 0, 1)
        data = torch.from_numpy(data)
        label = torch.LongTensor(np.asarray([label]))

        return data.float(), label