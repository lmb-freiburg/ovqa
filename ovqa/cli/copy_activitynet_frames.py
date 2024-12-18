"""
Call the extract_frames script with the ActivityNet val set video ids and paths.

Requires ffmpeg-python and system ffmpeg.
"""
import os
import shutil

from ovqa.paths import get_ovqa_annotations_dir
from packg.iotools import yield_lines_from_file
from ovqa.paths import get_data_dir
from packg.tqdmext import tqdm_max_ncols


def main():
    data_dir = get_data_dir()
    print(f"Data source dir: {data_dir}")
    input_path = data_dir / "activitynet" / "frames_uncropped_all"
    output_path = data_dir / "activitynet" / "frames_uncropped"
    frames_list_file = get_ovqa_annotations_dir() / "activitynet" / "frame_paths.txt"
    frames_list = list(yield_lines_from_file(frames_list_file))
    print(f"Frames list file: {frames_list_file} with {len(frames_list)} lines")

    for frame_file_rel in tqdm_max_ncols(frames_list, desc="Check frames exist"):
        frame_file_in = input_path / frame_file_rel
        if not frame_file_in.is_file():
            raise FileNotFoundError(f"Frame file {frame_file_in} does not exist")

    os.makedirs(output_path, exist_ok=True)
    for frame_file_rel in tqdm_max_ncols(frames_list, desc="Copy frames"):
        frame_file_in = input_path / frame_file_rel
        frame_file_out = output_path / frame_file_rel
        os.makedirs(frame_file_out.parent, exist_ok=True)
        shutil.copy(frame_file_in, frame_file_out)

    print(f"Done filling {output_path}")


if __name__ == "__main__":
    main()
