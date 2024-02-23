import os

from attrs import define
from loguru import logger
from pytube import YouTube
from pytube.cli import download_highest_resolution_progressive

from ovqa.paths import get_ovqa_annotations_dir
from packg.iotools import load_json
from packg.log import SHORTEST_FORMAT, configure_logger, get_logger_level_from_args
from packg.paths import get_data_dir
from typedparser import VerboseQuietArgs, TypedParser


@define
class Args(VerboseQuietArgs):
    pass


def main():
    parser = TypedParser.create_parser(Args, description=__doc__)
    args: Args = parser.parse_args()
    configure_logger(level=get_logger_level_from_args(args), format=SHORTEST_FORMAT)
    target_dir = get_data_dir() / "activitynet" / "videos"
    os.makedirs(target_dir, exist_ok=True)
    video_ids = load_json(get_ovqa_annotations_dir() / "activitynet/activitynet_val_video_ids.json")
    logger.info(f"Downloading {len(video_ids)} videos to {target_dir}")
    logger.info(f"Starting download...")
    for i, video_id in enumerate(video_ids):
        if i % 100 == 0:
            logger.info(f"Downloading video {i + 1}/{len(video_ids)}")
        yt = YouTube(f"http://youtube.com/watch?v={video_id}")
        download_highest_resolution_progressive(youtube=yt, resolution="highest", target=".")


if __name__ == "__main__":
    main()
