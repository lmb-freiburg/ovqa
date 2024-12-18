import os
import pytube.exceptions as exceptions
import sys
from attrs import define
from loguru import logger
from pytube import Stream, YouTube
from pytube.cli import download_highest_resolution_progressive as orig_dl, on_progress
from typing import Optional

from ovqa.paths import get_ovqa_annotations_dir
from packg import format_exception
from packg.iotools import load_json, yield_lines_from_file
from packg.log import SHORTEST_FORMAT, configure_logger, get_logger_level_from_args
from ovqa.paths import get_data_dir
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
    video_ids = list(yield_lines_from_file(
        get_ovqa_annotations_dir() / "activitynet/activitynet_val_video_ids.txt"
    ))
    logger.info(f"Downloading {len(video_ids)} videos to {target_dir}")
    logger.info(f"Starting download...")
    for i, video_id in enumerate(video_ids):
        if i % 100 == 0:
            logger.info(f"Downloading video {i + 1}/{len(video_ids)}")
        for k in range(3):
            try:
                download_highest_resolution_progressive(video_id, target=target_dir)
            except Exception as e:
                logger.error(f"Error downloading video {video_id}: {format_exception(e)}")
                continue
            break


_ = orig_dl  # reference


def download_highest_resolution_progressive(video_id: str, target: Optional[str] = None) -> None:
    youtube = YouTube(f"http://youtube.com/watch?v={video_id}")
    youtube.register_on_progress_callback(on_progress)
    try:
        stream = youtube.streams.get_highest_resolution()
    except exceptions.VideoUnavailable as err:
        print(f"No video streams available: {err}")
    else:
        try:
            _download(video_id, stream, target=target)
        except KeyboardInterrupt:
            sys.exit()


def _download(
    video_id: str,
    stream: Stream,
    target: Optional[str] = None,
) -> None:
    filesize_megabytes = stream.filesize // 1048576
    filename_orig = stream.default_filename
    ending = filename_orig.split(".")[-1]
    filename = f"v_{video_id}.{ending}"
    print(f"{filename} | {filesize_megabytes} MB | {filename_orig[:-len(ending)-1]}")
    file_path = stream.get_file_path(filename=filename, output_path=target)
    if stream.exists_at_path(file_path):
        print(f"Already downloaded at:\n{file_path}")
        return

    stream.download(output_path=target, filename=filename)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
