"""
Extract frames from videos.

Requires ffmpeg-python and system ffmpeg.
"""

from collections import OrderedDict

import json
import numpy as np
import os
import shutil
from attrs import define
from fractions import Fraction
from loguru import logger
from pathlib import Path
from typing import Optional

from ovqa.videos_to_frames_utils import (
    systemcall,
    MultiProcessor,
    get_video_ffprobe_info,
    get_video_info_from_ffprobe_result,
    convert_actions_to_ffmpeg_filter,
    get_center_crop,
    get_scaled_crop,
    get_smart_crop_resize_actions,
)
from packg.log import SHORTEST_FORMAT, configure_logger, get_logger_level_from_args
from packg.multiproc import FnMultiProcessor
from typedparser import add_argument, TypedParser, VerboseQuietArgs

FRAME_FILE = "frame_%010d.jpg"
FILETYPES = ["mp4", "mkv", "webm"]
FFPROBE_INFO_FILE = "ffprobe_videos.json"
FFPROBE_ANALYSIS_FILE = "ffprobe_results.txt"


@define(slots=False)
class Args(VerboseQuietArgs):
    input_path: str = add_argument(shortcut="-i", help="input path to videos", required=True)
    input_list: Optional[str] = add_argument(
        shortcut="-l", help="list of video file names to process less"
    )
    output_path: str = add_argument("output_path", help="output path")
    write: bool = add_argument("--write", action="store_true", help="Start the crop.")
    fps: float = add_argument("--fps", type=float, default=16, help="Frames per second.")
    num_frames: Optional[int] = add_argument(
        "--num_frames", type=int, help="Overwrite FPS and fix number of frames per video."
    )
    reload: bool = add_argument(
        "--reload",
        action="store_true",
        help="reload all video information from the files using ffmpeg",
    )
    width: int = add_argument(
        shortcut="-x", type=int, default=256, help="target width of extracted frames"
    )
    height: int = add_argument(
        shortcut="-y", type=int, default=256, help="target height of extracted frames"
    )
    quality: int = add_argument(type=int, default=2, help="frame jpeg quality (2=best, 31=worst)")
    num_workers: int = add_argument(
        type=int, default=0, help="how many processes to spawn, 0 means use cpu_count, "
    )
    max_videos: int = add_argument(
        type=int, default=-1, help="how many videos to process in one run, default -1 = all"
    )
    disable_progressbar: bool = add_argument(
        action="store_true", help="multiprocessing without progressbar"
    )
    crop_method: str = add_argument(
        type=str,
        default="scaled_crop",
        help="1) center_crop for a simple center crop or "
        "2) scaled_crop to crop the longer edge first and then resize the "
        "image to target size or 3) smart_crop 4) scale_shorter_side for no cropping",
    )
    smart_crop_ratio: float = add_argument("--smart_crop_ratio", type=float, default=0.5)
    smart_resize_min: float = add_argument("--smart_resize_min", type=float, default=1.1)
    smart_resize_max: float = add_argument("--smart_resize_max", type=float, default=1.25)


def main():
    parser = TypedParser.create_parser(Args, description=__doc__)
    args: Args = parser.parse_args()
    configure_logger(level=get_logger_level_from_args(args), format=SHORTEST_FORMAT)
    logger.info(f"{args}")
    extract_frames(args)


def extract_frames(args: Args):
    if args.crop_method == "scale_shorter_side":
        assert args.width == args.height, (
            f"For scale_shorter_side, x and y must be the same but are "
            f"{args.width} and {args.height}"
        )

    # Get list of videos
    input_path = Path(args.input_path)
    if not args.input_list:
        files = os.listdir(input_path)
    else:
        # if a list of files is given make sure all files are found
        all_files = os.listdir(input_path)
        files = [
            a
            for a in map(
                str.strip, Path(args.input_list).read_text("utf8").splitlines(keepends=False)
            )
            if a != ""
        ]
        fail = 0
        new_files = []
        for file in files:
            try_files = []
            for ending in ["", ".mp4", ".mkv", ".webm"]:
                try_files += [f"{file}{ending}", f"v_{file}{ending}"]

            for try_file in try_files:
                if try_file in all_files:
                    new_files.append(try_file)
                    break
            else:
                print(f"WARN: {file} not found in {all_files[:min(10, len(all_files))]} ...")
                fail += 1
        if fail > 0:
            print(f"WARN: Missing {fail} videos out of {len(files)}")
        files = new_files
        print(f"Got files: {files[:min(10, len(files))]} etc")
    file_keys = []
    file_formats = []

    # if max_videos is specified (e.g. for testing) only keep this amount of files
    if args.max_videos > 0:
        files = files[: args.max_videos]
        print(f"Only process first {args.max_videos} videos.")

    # Loop videos and determine file format
    for file in files:
        file_split = file.split(".")
        file_name = ".".join(file_split[:-1])
        file_type = file_split[-1]
        if (input_path / file).is_dir():
            print(f"SKIP: directory {file}")
            continue
        if file_type not in FILETYPES:
            print(f"SKIP: don't understand filetype of {file}")
            continue
        key = file_name
        file_keys.append(key)
        file_formats.append(file_type)

    # Make sure there is only one video per ID
    if len(list(set(file_keys))) != len(file_keys):
        # now there are multiple formats of the same video, keep only the first one
        new_file_keys, new_file_formats = [], []
        for key, formt in zip(file_keys, file_formats):
            if key in new_file_keys:
                continue
            new_file_keys.append(key)
            new_file_formats.append(formt)
        file_keys, file_formats = new_file_keys, new_file_formats
        print(f"Reduced to {len(file_keys)} videos.")

    # Create output path
    output_path = Path(args.output_path)
    os.makedirs(output_path, exist_ok=True)

    # Load video ffprobe info
    num_tasks = len(file_keys)
    print("reading ffprobe info of {} videos...".format(num_tasks))
    ffprobe_file = output_path / FFPROBE_INFO_FILE
    if not ffprobe_file.exists() or args.reload:
        # # single process for debugging
        # ffprobe_infos = {}
        # for file_key, file_format in zip(file_keys, file_formats):
        #     file_video = input_path / f"{file_key}.{file_format}"
        #     try:
        #         vid_id, ffprobe_info = ffmpeg.probe(str(file_video))
        #     except ffmpeg.Error as e:
        #         print("OUT", e.stdout)
        #         print("ERR", e.stderr)
        #         raise e
        #     ffprobe_infos[vid_id] = ffprobe_info
        #     print(ffprobe_info)
        #     # num_tasks -= 1

        # setup multiprocessing
        num_tasks = len(file_keys)
        mp = FnMultiProcessor(
            workers=args.num_workers,
            target_fn=run_ffprobe_fn,
            ignore_errors=True,
            verbose=not args.disable_progressbar,
            with_output=True,
            total=num_tasks,
            desc="Running ffprobe",
        )

        # enqueue tasks and run them
        for file_key, file_format in zip(file_keys, file_formats):
            file_video = input_path / f"{file_key}.{file_format}"
            mp.put(file_key, file_video)
        mp.run()
        ffprobe_infos = OrderedDict()
        print(f"\nStart reading output...")
        for _ in range(num_tasks):
            r = mp.get()
            if r is None:
                continue
            vid_id, ffprobe_info = r
            ffprobe_infos[vid_id] = ffprobe_info
        mp.close()
        # store to file
        with ffprobe_file.open("wt", encoding="utf8") as fh:
            json.dump(ffprobe_infos, fh, indent=4, sort_keys=True)
        print("wrote ffprobe info to: {:s}".format(str(ffprobe_file)))
    else:
        # reload from file
        with ffprobe_file.open("rt", encoding="utf8") as fh:
            ffprobe_infos = json.load(fh)
            print(f"Reloaded {len(ffprobe_infos)} videos from ffprobe results")
    print(f"{len(ffprobe_infos)} videos in ffprobe infos. {len(file_keys)} files to process.")
    ffprobe_infos_keys = set(ffprobe_infos.keys())
    missing_ffprobe_files = []
    for file in file_keys:
        if file not in ffprobe_infos_keys:
            missing_ffprobe_files.append(file)
    if len(missing_ffprobe_files) > 0:
        missing_files_str = "\n".join(missing_ffprobe_files)
        raise RuntimeError(
            f"Found {len(ffprobe_infos_keys)} files in ffprobe infos. Missing "
            f"{len(missing_ffprobe_files)} files:\n\n{missing_files_str}\n\n"
            f"Delete corrupt videos and re-run this script with --reload"
        )

    # analyze ffprobe info
    format_list, ratio_list, fps_list, duration_list = [], [], [], []
    for vid_id, ffprobe_info in ffprobe_infos.items():
        width, height, fps, duration = get_video_info_from_ffprobe_result(ffprobe_info)
        format_list.append((width, height))
        ratio_list.append(width / height)
        fps_list.append(fps)
        duration_list.append(duration)

    # print ffprobe info
    duration_list = np.array(duration_list)
    # noinspection PyArgumentList
    print(
        f"Durations (sec): min {duration_list.min():.3f}, max {duration_list.max():.3f}, "
        f"avg {duration_list.mean():.3f}, std {duration_list.std():.3f}"
    )
    ffprobe_analysis_file = output_path / FFPROBE_ANALYSIS_FILE
    with ffprobe_analysis_file.open("wt", encoding="utf8") as fh:
        print()
        format_list, ratio_list, fps_list = (
            sorted(list(set(a))) for a in (format_list, ratio_list, fps_list)
        )
        ratio_list = [float("{:.3f}".format(a)) for a in ratio_list]
        fps_list = [float("{:.3f}".format(float(Fraction(a)))) for a in fps_list]
        for file_h in [fh]:
            print(
                "formats: {}\nratios (w/h): {}\nframerates: {}".format(
                    format_list, ratio_list, fps_list
                ),
                file=file_h,
            )
        formats_x, formats_y = zip(*format_list)
        for item, name in zip(
            [formats_x, formats_y, ratio_list, fps_list],
            "formats_x, formats_y, ratio_list, fps_list".split(", "),
        ):
            hist, bin_edges = np.histogram(item, bins=30)
            bin_means = bin_edges[:-1] + (bin_edges[1:] - bin_edges[:-1]) / 2
            print("-" * 20, name)
            print_table = list((f"{m:5.0f}", f"{h:5.0f}") for h, m in zip(hist, bin_means))
            print(" ".join(t[0] for t in print_table))
            print(" ".join(t[1] for t in print_table))

    # see how many videos still need converting to frames
    done_file = output_path / "done.txt"
    done_keys = []
    file_keys_process = file_keys
    file_formats_process = file_formats
    if done_file.is_file():
        done_keys = done_file.read_text().splitlines()
        file_keys_process = []
        file_formats_process = []
        for file_key, file_format in zip(file_keys, file_formats):
            if file_key in done_keys:
                continue
            file_keys_process.append(file_key)
            file_formats_process.append(file_format)

    print(f"{len(done_keys)} already done, {len(file_keys_process)} left.")

    # for test only, exit here
    if not args.write:
        print(f"No --write so exiting.")
        return

    # start multiprocessing
    num_tasks = len(file_keys_process)
    mp = FnMultiProcessor(
        workers=args.num_workers,
        target_fn=extract_frames_fn,
        ignore_errors=True,
        verbose=not args.disable_progressbar,
        with_output=True,
        total=num_tasks,
        desc="Extracting frames",
    )

    # enqueue tasks and run
    for file_key, file_format in zip(file_keys_process, file_formats_process):
        # determine video file path and frame folder
        file_video_full = input_path / f"{file_key}.{file_format}"
        path_frames_full = output_path / file_key

        # read ffprobe info
        ffprobe_info = ffprobe_infos[file_key]

        # create task and enqueue it
        frames_per_video = None
        if isinstance(args.num_frames, int) and args.num_frames > 0:
            frames_per_video = args.num_frames
        mp.put(
            file_key,
            str(file_video_full),
            path_frames_full,
            ffprobe_info,
            args.width,
            args.height,
            args.fps,
            args.quality,
            args.crop_method,
            args.smart_crop_ratio,
            args.smart_resize_min,
            args.smart_resize_max,
            args.verbose,
            frames_per_video,
        )
    mp.run()

    # read results
    done_fh = done_file.open("wt")
    done_fh.write("\n".join(done_keys))
    print("analyzing results")
    for _ in range(num_tasks):
        result = mp.get()
        if result is None:
            # do not update data for failed videos
            continue
        else:
            vid_id, retcode, w, h, fps, num_frames = result
            done_fh.write(f"{vid_id}\n")

    # systemcall("stty sane")
    os.system("stty sane")


def run_ffprobe_fn(vid_id, file_video):
    probe_info = get_video_ffprobe_info(file_video)
    return vid_id, probe_info


def extract_frames_fn(
    vid_id,
    file_video,
    folder_frames,
    ffprobe_info,
    target_w,
    target_h,
    target_fps,
    qscale,
    crop_method="scaled_crop",
    smart_crop_ratio=0.5,
    smart_resize_min=1.1,
    smart_resize_max=1.25,
    verbose=False,
    frames_per_video: Optional[int] = None,
):
    # frames_per_video: this overrides fps such that there is a fixed amount of frames per video
    # get width and height from ffprobe info
    w, h, fps, duration = get_video_info_from_ffprobe_result(ffprobe_info)

    # prepare empty frame directory
    shutil.rmtree(str(folder_frames), ignore_errors=True)
    os.makedirs(str(folder_frames))

    if crop_method == "scaled_crop":
        # get scaled crop
        crop_y, crop_x, crop_h, crop_w = get_scaled_crop(h, w, target_h, target_w)
        ffmpeg_filter = "crop={:d}:{:d}:{:d}:{:d},scale={:d}:{:d}".format(
            crop_w, crop_h, crop_x, crop_y, target_w, target_h
        )
        # print(f"Scaled crop filter: {ffmpeg_filter}")
    elif crop_method == "center_crop":
        # get center crop
        crop_x, crop_y, crop_w, crop_h = get_center_crop(h, w, target_h, target_w)
        ffmpeg_filter = "crop={:d}:{:d}:{:d}:{:d},".format(crop_w, crop_h, crop_x, crop_y)
    elif crop_method == "smart_crop":
        # get smart crop actions
        actions = get_smart_crop_resize_actions(
            h,
            w,
            target_h,
            target_w,
            smart_crop_ratio,
            smart_resize_min,
            smart_resize_max,
            verbose=verbose,
        )
        ffmpeg_filter = convert_actions_to_ffmpeg_filter(actions)
    elif crop_method == "scale_shorter_side":
        assert (
            target_w == target_h
        ), f"For scale_shorter_side, x and y must be same but are {target_w} and {target_h}"
        target_scale = target_w
        if h <= target_scale or w <= target_scale:
            # the shorter side is already small enough
            ffmpeg_filter = None
        elif w < h:
            # width is the shorter side, scale that to target
            ffmpeg_filter = f"scale={target_scale}:-1"
        else:
            # height is the shorter side, scale that to target
            ffmpeg_filter = f"scale=-1:{target_scale}"
    else:
        raise ValueError("crop method not found: {}".format(crop_method))

    target_fps = target_fps
    if frames_per_video is not None:
        # without fps
        target_fps = frames_per_video / ffprobe_info["duration"]

    # define the ffmpeg command with filters
    file_frames = str(folder_frames / FRAME_FILE)
    if ffmpeg_filter is not None:
        filter_string = f'-vf "{ffmpeg_filter:s},fps={target_fps:f}"'
    else:
        filter_string = f'-vf "fps={target_fps:f}"'

    cmd = (
        f"ffmpeg -i {file_video:s} -hide_banner {filter_string} "
        f"-qscale:v {qscale:d} {file_frames:s}"
    )

    if verbose:
        print("command:", cmd)

    # run command
    out, err, retcode = systemcall(cmd)
    if retcode != 0:
        print()
        print("WARNING: video {} failed with return code {}".format(vid_id, retcode))
        print("command was: {}".format(cmd))
        print("stdout:", out)
        print("stderr:", err)
        raise RuntimeError("video processing for {} failed, see stdout".format(vid_id))

    # check how many frames where created
    num_frames = len(os.listdir(str(folder_frames)))

    return vid_id, retcode, w, h, fps, num_frames


if __name__ == "__main__":
    main()
