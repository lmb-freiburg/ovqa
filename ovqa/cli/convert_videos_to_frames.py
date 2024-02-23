"""
python -m vislan.run.data_videos_to_frames -x 480 -y 480 --crop_method scale_shorter_side \
    -i /misc/lmbssd/gings/datasets/activitynet/raw/videos \
    /misc/lmbssd/gings/datasets/activitynet/raw/frames_uncropped \
    -l /misc/lmbssd/gings/datasets/activitynet/raw/video_list_subset_validation.txt
"""

import argparse
import json
import os
import shutil
from collections import OrderedDict
from fractions import Fraction
from pathlib import Path
from typing import Optional

import numpy as np

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

FRAME_FILE = "frame_%010d.jpg"
FILETYPES = ["mp4", "mkv", "webm"]
FFPROBE_INFO_FILE = "ffprobe_videos.json"
FFPROBE_ANALYSIS_FILE = "ffprobe_results.txt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input_path", type=str, help="input path to videos", default=None, required=True
    )
    parser.add_argument(
        "-l",
        "--input_list",
        type=str,
        help="list of video file names to process less",
        default=None,
    )
    parser.add_argument("output_path", type=str, help="output path")
    parser.add_argument("--write", action="store_true", help="Start the crop.")
    parser.add_argument("--fps", type=float, default=16, help="Frames per second.")
    parser.add_argument(
        "--num_frames",
        type=int,
        default=None,
        help="Overwrite FPS and fix number of frames per video.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="reload all video information from the files using ffmpeg",
    )
    parser.add_argument(
        "-x", "--width", type=int, default=256, help="target width of extracted frames"
    )
    parser.add_argument(
        "-y", "--height", type=int, default=256, help="target height of extracted frames"
    )
    parser.add_argument(
        "-q", "--quality", type=int, default=2, help="frame jpeg quality (2=best, 31=worst)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="how many processes to spawn, 0 means use cpu_count, ",
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=-1,
        help="how many videos to process in one run, default -1 = all",
    )
    parser.add_argument(
        "--disable_progressbar", action="store_true", help="multiprocessing without progressbar"
    )
    parser.add_argument(
        "--crop_method",
        type=str,
        default="scaled_crop",
        help="1) center_crop for a simple center crop or "
        "2) scaled_crop to crop the longer edge first and then resize the "
        "image to target size or 3) smart_crop 4) scale_shorter_side for no cropping",
    )
    parser.add_argument("--smart_crop_ratio", type=float, default=0.5)
    parser.add_argument("--smart_resize_min", type=float, default=1.1)
    parser.add_argument("--smart_resize_max", type=float, default=1.25)
    parser.add_argument("--verbose", action="store_true", help="more output")
    args = parser.parse_args()

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
        if fail:
            raise ValueError(f"Missing {fail} videos out of {len(files)}")
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
        mp = MultiProcessor(num_workers=args.num_workers, progressbar=not args.disable_progressbar)

        # enqueue tasks and run them
        num_tasks = len(file_keys)
        for file_key, file_format in zip(file_keys, file_formats):
            file_video = input_path / f"{file_key}.{file_format}"
            mp.add_task(TaskFfprobe(file_key, file_video))
        results = mp.run()
        mp.close()

        # read results
        ffprobe_infos = OrderedDict()
        for r in results:
            vid_id, ffprobe_info = r
            ffprobe_infos[vid_id] = ffprobe_info
            num_tasks -= 1

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
    assert all(
        file in ffprobe_infos for file in file_keys
    ), f"{ffprobe_infos.keys()}, {file_keys}. FFPROBE info seems incorrect, try reloading with --reload"

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
    mp = MultiProcessor(num_workers=args.num_workers, progressbar=not args.disable_progressbar)

    # enqueue tasks and run
    num_tasks = len(file_keys_process)
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
        task = TaskExtractFrames(
            file_key,
            str(file_video_full),
            path_frames_full,
            ffprobe_info,
            args.width,
            args.height,
            args.fps,
            args.quality,
            crop_method=args.crop_method,
            verbose=args.verbose,
            frames_per_video=frames_per_video,
        )
        mp.add_task(task)
    results = mp.run()
    # mp.close()

    # read results
    done_fh = done_file.open("wt")
    done_fh.write("\n".join(done_keys))
    print("analyzing results")
    for result in results:
        if result is None:
            # do not update data for failed videos
            num_tasks -= 1
            continue
        else:
            num_tasks -= 1
            vid_id, retcode, w, h, fps, num_frames = result
            done_fh.write(f"{vid_id}\n")

    # systemcall("stty sane")
    os.system("stty sane")


class TaskFfprobe(object):
    """
    Video analysis with ffprobe
    """

    def __init__(self, vid_id, file_video):
        self.vid_id = vid_id
        self.file_video = file_video

    def __call__(self):
        probe_info = get_video_ffprobe_info(self.file_video)
        # DEBUG PRINT
        # print(f"Duration is {probe_info['duration']}")
        return self.vid_id, probe_info

    def __str__(self):
        return "ffmpeg.probe on video {:s}".format(self.vid_id)


class TaskExtractFrames(object):
    """
    Frame extraction with ffmpeg
    """

    def __init__(
        self,
        vid_id,
        file_video,
        folder_frames,
        ffprobe_info,
        tw,
        th,
        fps,
        qscale,
        crop_method="scaled_crop",
        smart_crop_ratio=0.5,
        smart_resize_min=1.1,
        smart_resize_max=1.25,
        assume_done=False,
        verbose=False,
        frames_per_video: Optional[int] = None,
    ):
        """

        Args:
            vid_id:
            file_video:
            folder_frames:
            ffprobe_info:
            tw:
            th:
            fps:
            qscale:
            crop_method:
            smart_crop_ratio:
            smart_resize_min:
            smart_resize_max:
            assume_done:
            verbose:
            frames_per_video: this overrides fps such that there is a fixed amount of frames per video
        """
        self.vid_id = vid_id
        self.file_video = file_video
        self.folder_frames = folder_frames
        self.ffprobe_info = ffprobe_info
        self.target_w = tw
        self.target_h = th
        self.target_fps = fps
        self.qscale = qscale
        self.crop_method = crop_method
        self.smart_crop_ratio = smart_crop_ratio
        self.smart_resize_min = smart_resize_min
        self.smart_resize_max = smart_resize_max
        self.assume_done = assume_done
        self.verbose = verbose
        self.frames_per_video = frames_per_video

    def __call__(self):
        # get width and height from ffprobe info
        w, h, fps, duration = get_video_info_from_ffprobe_result(self.ffprobe_info)
        target_w, target_h = self.target_w, self.target_h

        # prepare empty frame directory
        shutil.rmtree(str(self.folder_frames), ignore_errors=True)
        os.makedirs(str(self.folder_frames))

        if self.crop_method == "scaled_crop":
            # get scaled crop
            crop_y, crop_x, crop_h, crop_w = get_scaled_crop(h, w, target_h, target_w)
            ffmpeg_filter = "crop={:d}:{:d}:{:d}:{:d},scale={:d}:{:d}".format(
                crop_w, crop_h, crop_x, crop_y, target_w, target_h
            )
            # print(f"Scaled crop filter: {ffmpeg_filter}")
        elif self.crop_method == "center_crop":
            # get center crop
            crop_x, crop_y, crop_w, crop_h = get_center_crop(h, w, target_h, target_w)
            ffmpeg_filter = "crop={:d}:{:d}:{:d}:{:d},".format(crop_w, crop_h, crop_x, crop_y)
        elif self.crop_method == "smart_crop":
            # get smart crop actions
            actions = get_smart_crop_resize_actions(
                h,
                w,
                target_h,
                target_w,
                self.smart_crop_ratio,
                self.smart_resize_min,
                self.smart_resize_max,
                verbose=self.verbose,
            )
            ffmpeg_filter = convert_actions_to_ffmpeg_filter(actions)
        elif self.crop_method == "scale_shorter_side":
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
            raise ValueError("crop method not found: {}".format(self.crop_method))

        target_fps = self.target_fps
        if self.frames_per_video is not None:
            # without fps
            target_fps = self.frames_per_video / self.ffprobe_info["duration"]

        # define the ffmpeg command with filters
        file_frames = str(self.folder_frames / FRAME_FILE)
        if ffmpeg_filter is not None:
            filter_string = f'-vf "{ffmpeg_filter:s},fps={target_fps:f}"'
        else:
            filter_string = f'-vf "fps={target_fps:f}"'

        cmd = (
            f"ffmpeg -i {self.file_video:s} -hide_banner {filter_string} "
            f"-qscale:v {self.qscale:d} {file_frames:s}"
        )

        if self.verbose:
            print("command:", cmd)

        # run command
        out, err, retcode = systemcall(cmd)
        if retcode != 0:
            print()
            print("WARNING: video {} failed with return code {}".format(self.vid_id, retcode))
            print("command was: {}".format(cmd))
            print("stdout:", out)
            print("stderr:", err)
            raise RuntimeError("video processing for {} failed, see stdout".format(self.vid_id))

        # check how many frames where created
        num_frames = len(os.listdir(str(self.folder_frames)))

        return self.vid_id, retcode, w, h, fps, num_frames


if __name__ == "__main__":
    main()
