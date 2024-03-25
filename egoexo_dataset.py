import numpy as np
import torch
import os
import PIL
from typing import Tuple
import re

import torch
import torchvision.transforms as transforms
from utils import read_pfm


class EgoExoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
        train_val_split: float = 0.8,
        subsampling_rate: float = 1.0,
        split: str = "train",
        seed: int = 10,
        img_size: int = 128,
        random_split: bool = False,
        scenario_excluding_list: list = ["cooking"],
    ):
        assert train_val_split >= 0.0 and train_val_split <= 1.0
        assert subsampling_rate >= 0.0 and subsampling_rate <= 1.0
        assert split in ["train", "test"]
        self.data_dir = data_dir
        self.train_val_split = train_val_split
        self.subsampling_rate = subsampling_rate
        self.split = split
        self._rng = np.random.default_rng(seed=seed)
        self.seed = seed
        self.img_size = img_size
        self.random_split = random_split
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.scenario_excluding_list = scenario_excluding_list

        self.cam_map = {}

        self.take_list = self.load_takes(data_dir)
        self.take_list = self.subsample_takes(self.take_list)

    def load_takes(self, data_dir):
        # TODO: try your best to limit hardcoding
        take_list = []
        for take_name in os.listdir(data_dir):
            try:
                take_path = os.path.join(data_dir, take_name, "frames")
                if not os.path.isdir(take_path):
                    continue
                cameras = os.listdir(take_path)
                cameras.sort()
                self.cam_map[take_name] = cameras
                num_cameras = len(cameras)
                num_frames = len(os.listdir(os.path.join(take_path, "cam01")))
                if take_name.split("_")[1] in self.scenario_excluding_list:
                    continue
                take_list.append((take_name, num_cameras, num_frames))
            except:
                print(f"Error loading take {take_name}")
                continue

        if self.random_split:
            self._rng.shuffle(take_list)
        take_list = (
            take_list[: int(len(take_list) * self.train_val_split)]
            if self.split == "train"
            else take_list[int(len(take_list) * self.train_val_split) :]
        )
        return take_list

    def subsample_takes(self, take_list):
        # subsample episodes
        if self.subsampling_rate < 1.0:
            self._rng.choice(
                take_list,
                size=int(len(take_list) * self.subsampling_rate),
                replace=False,
            )
        return take_list

    def with_transform(self, transform):
        self.transform = transform

    def get_camera_pose(self, cam_pose):
        cam2world = np.array(cam_pose[:15]).reshape(3, 5)

        # correct the rotation matrix order: llff to nerf
        cam2world = np.concatenate(
            [
                cam2world[:, 1:2],
                -cam2world[:, 0:1],
                cam2world[:, 2:3],
                cam2world[:, 3:],
            ],
            axis=1,
        )

        extrinsic_mat = cam2world[:3, :4]
        extrinsic_mat = np.concatenate(
            [extrinsic_mat, np.array([[0, 0, 0, 1]])], axis=0
        )  # (4, 4)

        h, w, f = cam2world[:, -1]
        intrinsic_mat = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
        return extrinsic_mat, intrinsic_mat

    def __len__(self):
        return len(self.take_list)

    def __getitem__(self, index):
        # TODO: fix this hardcoding
        if self.split == "test":
            self._rng = np.random.default_rng(seed=self.seed)
        take_name, num_cameras, num_frames = self.take_list[index]
        pose_camera_idx = self._rng.integers(num_cameras) + 1
        # timestep_camera_idx = (
        #     self._rng.integers(pose_camera_idx - 1, pose_camera_idx + 1) % num_cameras
        #     + 1
        # )
        timestep_camera_idx = self._rng.integers(num_cameras) + 1
        pose_frame_idx = self._rng.integers(num_frames) + 1
        # timestep_frame_idx = self._rng.choice(
        #     [
        #         i
        #         for i in list(
        #             range(max(pose_frame_idx - 20, 0), max(pose_frame_idx - 5, 0))
        #         )
        #         + list(
        #             range(
        #                 min(pose_frame_idx + 5, num_frames),
        #                 min(pose_frame_idx + 20, num_frames),
        #             )
        #         )
        #     ]
        # ) + 1
        timestep_frame_idx = self._rng.integers(num_frames) + 1

        cam_poses_path = os.path.join(
            self.data_dir,
            take_name,
            "poses_bounds.npy",
        )
        cam_poses = np.load(cam_poses_path)
        source_extrinsics, source_intrinsics = self.get_camera_pose(
            cam_poses[pose_camera_idx - 1]
        )
        target_extrinsics, target_intrinsics = self.get_camera_pose(
            cam_poses[timestep_camera_idx - 1]
        )

        pose_cam = self.cam_map[take_name][pose_camera_idx - 1]
        timestep_cam = self.cam_map[take_name][timestep_camera_idx - 1]

        pose_frame_path = os.path.join(
            self.data_dir,
            take_name,
            "frames",
            pose_cam,
            f"frame{pose_frame_idx:04d}.jpg",
        )
        timestep_frame_path = os.path.join(
            self.data_dir,
            take_name,
            "frames",
            timestep_cam,
            f"frame{timestep_frame_idx:04d}.jpg",
        )
        target_frame_path = os.path.join(
            self.data_dir,
            take_name,
            "frames",
            timestep_cam,
            f"frame{pose_frame_idx:04d}.jpg",
        )

        pose_depth_path = os.path.join(
            self.data_dir,
            take_name,
            "frames",
            pose_cam,
            "depth_est",
            f"mvs_depth_frame{pose_frame_idx:04d}.pfm",
        )
        pose_depth, scale = read_pfm(pose_depth_path)
        assert scale == 1.0
        pose_depth = pose_depth.squeeze()  # [H, W]
        min_depth, max_depth = np.percentile(pose_depth, 5), np.percentile(
            pose_depth, 95
        )
        pose_depth[pose_depth < min_depth] = min_depth
        pose_depth[pose_depth > max_depth] = max_depth

        sample_dict = {
            "input_pose_image": PIL.Image.open(pose_frame_path),
            "input_pose_depth": pose_depth,
            "input_pose_extrinsics": source_extrinsics,
            "input_pose_intrinsics": source_intrinsics,
            "input_timestep_extrinsics": target_extrinsics,
            "input_timestep_intrinsics": target_intrinsics,
            "input_timestep_image": PIL.Image.open(timestep_frame_path),
            "edited_image": PIL.Image.open(target_frame_path),
        }
        # print(pose_depth.shape)
        # print(sample_dict["input_pose_image"].size)
        # input()
        return self.transform(sample_dict)
