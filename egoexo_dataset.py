import numpy as np
import torch
import os
import PIL

import torch
import torchvision.transforms as transforms


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

    def __len__(self):
        return len(self.take_list)

    def __getitem__(self, index):
        # TODO: fix this hardcoding
        if self.split == "test":
            self._rng = np.random.default_rng(seed=self.seed)
        take_name, num_cameras, num_frames = self.take_list[index]
        pose_camera_idx = self._rng.integers(num_cameras) + 1
        timestep_camera_idx = (
            self._rng.integers(pose_camera_idx - 1, pose_camera_idx + 1) % num_cameras
            + 1
        )
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
            pose_cam,
            f"frame{timestep_frame_idx:04d}.jpg",
        )

        sample_dict = {
            "input_pose_image": PIL.Image.open(pose_frame_path),
            "input_timestep_image": PIL.Image.open(timestep_frame_path),
            "edited_image": PIL.Image.open(target_frame_path),
        }
        return self.transform(sample_dict)
