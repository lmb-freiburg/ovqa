"""
Call the extract_frames script with the ActivityNet val set video ids and paths.

Requires ffmpeg-python and system ffmpeg.
"""
import argparse
from pprint import pprint

from ovqa.cli.convert_videos_to_frames import extract_frames, Args as ExtractFramesArgs
from ovqa.paths import get_ovqa_annotations_dir
from packg.paths import get_data_dir
from typedparser import attrs_from_dict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="how many processes to spawn, 0 means use cpu_count.",
    )
    args = parser.parse_args()
    data_dir = get_data_dir()
    print(f"Data source dir: {data_dir}")
    attr_dict = dict(
        input_path=data_dir / "activitynet" / "raw/videos",
        output_path=data_dir / "activitynet" / "frames_uncropped_all",
        input_list=get_ovqa_annotations_dir() / "activitynet" / "activitynet_val_video_ids.txt",
        width=480,
        height=480,
        crop_method="scale_shorter_side",
        write=True,
        num_workers=args.num_workers,
    )
    print(f"Script arguments:")
    pprint(attr_dict)
    args = attrs_from_dict(ExtractFramesArgs, attr_dict)
    print(f"Args instance: {args}")
    extract_frames(args)


if __name__ == "__main__":
    main()
