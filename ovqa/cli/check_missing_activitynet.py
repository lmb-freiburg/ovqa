import os
from attrs import define
from loguru import logger

from ovqa.paths import get_ovqa_annotations_dir
from packg.iotools import yield_lines_from_file
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
    video_ids = list(yield_lines_from_file(
        get_ovqa_annotations_dir() / "activitynet/activitynet_val_video_ids.txt"
    ))
    logger.info(f"Looking for {len(video_ids)} videos in {target_dir}")
    files = os.listdir(target_dir)
    found_dict = {video_id:False for video_id in video_ids}
    for file in files:
        full_file = target_dir / file
        if not full_file.is_file():
            continue
        name_only = ".".join(file.split(".")[:-1])
        if name_only in found_dict:
            found_dict[name_only] = True
    found_list, missing_list = [], []
    for video_id, found in found_dict.items():
        if found:
            found_list.append(video_id)
        else:
            missing_list.append(video_id)
    found_list = sorted(set(found_list))
    missing_list = sorted(set(missing_list))
    logger.info(f"Found {len(found_list)} videos, missing {len(missing_list)}")
    missing_list_file = get_data_dir() / "activitynet" / "missing_video_ids.txt"
    missing_list_file.write_text("\n".join(missing_list + [""]))
    logger.info(f"Wrote missing list to {missing_list_file.as_posix()}")


if __name__ == "__main__":
    main()
