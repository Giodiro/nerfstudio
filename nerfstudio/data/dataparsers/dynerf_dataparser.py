# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data parser for dynerf dataset"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Type

import numpy as np
import torch

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.dataparsers.base_dataparser import (
    DataparserOutputs,
)
from nerfstudio.data.dataparsers.nerfstudio_dataparser import Nerfstudio, NerfstudioDataParserConfig
from nerfstudio.utils.io import load_from_json


@dataclass
class DyNeRFDataParserConfig(NerfstudioDataParserConfig):
    """DyNeRF dataset parser config"""

    _target: Type = field(default_factory=lambda: DyNeRF)
    """target class to instantiate"""
    data: Path = Path("data/dynerf/flame_salmon")
    """Directory specifying location of data."""


class DyNeRF(Nerfstudio):
    """DyNeRF Dataset"""

    config: DyNeRFDataParserConfig

    def _generate_dataparser_outputs(self, split="train"):
        self.downscale_factor = self.config.downscale_factor or 1
        self.config.train_split_percentage = 1.0
        ns_outputs = super()._generate_dataparser_outputs("train")

        # Have to re-read the json meta information to extract cameras and timestamps
        # from each frame
        if self.config.data.suffix == ".json":
            meta = load_from_json(self.config.data)
            data_dir = self.config.data.parent
        else:
            meta = load_from_json(self.config.data / "transforms.json")
            data_dir = self.config.data
        time = []
        camera_idx = []
        for frame in meta["frames"]:
            filepath = Path(frame["file_path"])
            fname = self._get_fname(filepath, data_dir)
            if not fname.exists():
                continue
            time.append(frame["time"])
            camera_idx.append(frame["camera_idx"])

        time = torch.tensor(time, dtype=torch.float32)
        # Normalize time to be in 0, +1 (same as dnerf_dataparser)
        time = ((time - time.amin()) / time.amax())
        camera_idx = torch.tensor(camera_idx, dtype=torch.int64)

        # Choose train/test based on camera indices:
        # in DyNeRF: camera 0 is for testing, the others are for training
        num_images = len(ns_outputs.image_filenames)
        i_all = np.arange(num_images)
        i_train = i_all[camera_idx != 0]
        i_eval = np.setdiff1d(i_all, i_train)  # eval images are the remaining images
        if split == "train":
            indices = i_train
        elif split in ["val", "test"]:
            indices = i_eval
        else:
            raise ValueError(f"Unknown dataparser split {split}")
        idx_tensor = torch.tensor(indices, dtype=torch.long)  # TODO: Check if this can be avoided

        cameras = Cameras(
            fx=ns_outputs.cameras.fx[idx_tensor],
            fy=ns_outputs.cameras.fy[idx_tensor],
            cx=ns_outputs.cameras.cx[idx_tensor],
            cy=ns_outputs.cameras.cy[idx_tensor],
            distortion_params=ns_outputs.cameras.distortion_params[idx_tensor],  # TODO: Delete?
            height=ns_outputs.cameras.height[idx_tensor],
            width=ns_outputs.cameras.width[idx_tensor],
            camera_to_worlds=ns_outputs.cameras.camera_to_worlds[idx_tensor],
            camera_type=ns_outputs.cameras.camera_type[idx_tensor],
            times=time[idx_tensor],
        )
        dataparser_outputs = DataparserOutputs(
            image_filenames=[ns_outputs.image_filenames[i] for i in indices],
            cameras=cameras,
            scene_box=ns_outputs.scene_box,
            dataparser_scale=ns_outputs.dataparser_scale,
            dataparser_transform=ns_outputs.dataparser_transform,
        )

        return dataparser_outputs

    def _get_fname(self, filepath: PurePath, data_dir: Path, folder_prefix="images") -> Path:
        """Get the filename of the image file.

        filepath: the base file name of the transformations.
        data_dir: the directory of the data that contains the transform file
        folder_prefix: prefix of the newly generated downsampled images
        """
        dscale_factor = self.config.downscale_factor
        if dscale_factor is not None and dscale_factor > 1:
            return data_dir / f"{folder_prefix}_{dscale_factor}" / filepath
        return data_dir / f"{folder_prefix}" / filepath
