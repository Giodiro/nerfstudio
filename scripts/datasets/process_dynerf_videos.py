#!/usr/bin/env python
"""Processes a video sequence for use ."""
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional, Iterator
from typing_extensions import Annotated

import numpy as np
import tyro
from rich.console import Console

from nerfstudio.process_data import process_data_utils
from nerfstudio.utils import install_checks

CONSOLE = Console(width=120)


@dataclass
class ProcessVideo:
    data: Path
    """Path to the DyNeRF data: a directory of videos, each from a different pose."""
    output_dir: Path
    """Path to the output directory."""
    num_frames_target: Optional[int] = None
    """Target number of frames to use for the dataset, results may not be exact."""
    num_downscales: int = 3
    """Number of times to downscale the images. Downscales by 2 each time. For example a value of 3
        will downscale the images by 2x, 4x, and 8x."""
    verbose: bool = False
    """If True, print extra logging."""
    skip_image_extraction: bool = False
    """If True, skip video to image, and downsampling steps. Only poses are processed."""

    def main(self) -> None:
        """Process video into a nerfstudio dataset."""
        install_checks.check_ffmpeg_installed()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        image_dir = self.output_dir / "images"

        poses_bounds = np.load(str(self.data / 'poses_bounds.npy'))  # (n_cameras, 17)
        num_cameras = poses_bounds.shape[0]

        videopaths = np.array(list(self.data.glob("*.mp4")))
        videopaths.sort()
        assert len(videopaths) == num_cameras, \
            'Mismatch between number of cameras and number of poses!'
        CONSOLE.log(f"[green] Starting preprocessing of videos from {num_cameras} cameras.")

        frames = []
        summary_log = []
        for cam_id, videopath in enumerate(videopaths):
            camera_image_dir = image_dir / f"camera_{cam_id}"
            camera_image_dir.mkdir(parents=True, exist_ok=True)
            if not self.skip_image_extraction:
                summary_log, num_extracted_frames = process_data_utils.convert_video_to_images(
                    videopath, image_dir=camera_image_dir, num_frames_target=self.num_frames_target,
                    verbose=self.verbose
                )
                summary_log.append(
                    process_data_utils.downscale_images(
                        camera_image_dir, self.num_downscales, verbose=self.verbose,
                        folder_name=f"camera_{cam_id}",
                ))
                # Move downsampled images from data/images/camera_0_2/*.png to data/images_2/camera_0/
                downsample_factors = [2 ** i for i in range(1, self.num_downscales + 1)]
                for downsample_factor in downsample_factors:
                    downsample_path = image_dir / f"camera_{cam_id}_{downsample_factor}"
                    new_downsample_path = self.output_dir / f"images_{downsample_factor}" / f"camera_{cam_id}"
                    new_downsample_path.mkdir(parents=True, exist_ok=True)
                    downsample_path.replace(new_downsample_path)

            # Convert poses_bounds.npy to transforms.json
            cam_pose = poses_bounds[cam_id, :15].reshape(3, 5)
            near_far = poses_bounds[cam_id, -2:]  # TODO: ignored
            H, W, focal = cam_pose[:, -1]
            cam_pose = cam_pose[:, :-1]  # 3, 4
            # Original poses has rotation in form "down right back", change to "right up back"
            # See https://github.com/bmild/nerf/issues/34
            cam_pose = np.concatenate([cam_pose[:, 1:2], -cam_pose[:, :1], cam_pose[:, 2:4]], -1)
            # convert to homogeneous coordinates
            cam_pose = np.concatenate((cam_pose, np.array([[0, 0, 0, 1]])), 0)
            files: Iterator[os.DirEntry] = os.scandir(camera_image_dir)
            num_frames_in_cam = 0
            for f in files:  # Extracted images have file-name like frame_%05d.png
                if not f.is_file() or not f.name.endswith("png"):
                    continue
                frames.append({
                    "file_path": str(Path(f.path).relative_to(image_dir)),
                    "transform_matrix": cam_pose.tolist(),
                    "camera_idx": cam_id,
                    "time": int(re.match(r"frame_([0-9]+)\.png", f.name).group(1))
                })
                num_frames_in_cam += 1
            CONSOLE.log(f"[green]Extracted {num_frames_in_cam} images from camera {cam_id}")
            for summary in summary_log:
                CONSOLE.print(summary, justify="center")
            CONSOLE.rule()

        transforms = {
            "frames": frames,
            "fl_x": focal, "fl_y": focal,
            "cx": W / 2, "cy": H / 2,
            "w": W, "h": H,
        }
        with open(self.output_dir / "transforms.json", "w", encoding="UTF-8") as file:
            json.dump(transforms, file, indent=2)
        CONSOLE.rule("[bold green]:tada: :tada: :tada: All DONE :tada: :tada: :tada:")


Commands = Union[
    Annotated[ProcessVideo, tyro.conf.subcommand(name="video")],
]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()


if __name__ == "__main__":
    entrypoint()


# For sphinx docs
def get_parser_fn():
    tyro.extras.get_parser(Commands)
