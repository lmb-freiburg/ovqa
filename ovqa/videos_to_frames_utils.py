import ffmpeg
import multiprocessing
import numpy as np
import subprocess
import time
import tqdm
import traceback
from pathlib import Path
from timeit import default_timer as timer
from typing import Tuple, List, Sequence, Union

from packg import format_exception


def systemcall(call: Union[str, Sequence[str]]):
    pipe = subprocess.PIPE
    process = subprocess.Popen(call, stdout=pipe, stderr=pipe, shell=True)
    out, err = process.communicate()
    retcode = process.poll()
    charset = "utf-8"
    out = out.decode(charset)
    err = err.decode(charset)
    return out, err, retcode


class Worker(multiprocessing.Process):
    def __init__(self, task_q, result_q, error_q, verbose=False):
        super().__init__()
        self.task_q = task_q
        self.result_q = result_q
        self.error_q = error_q
        self.verbose = verbose

    def run(self):
        proc_name = self.name
        while True:
            next_task = self.task_q.get()
            if next_task is None:
                # poison pill means shutdown
                if self.verbose:
                    print("{:s}: exiting".format(proc_name))
                self.task_q.task_done()
                break
            if self.verbose:
                print(str(next_task))
            try:
                result = next_task()
                pass
            except Exception as e:
                self.error_q.put((e, traceback.format_exc()))
                result = None
            self.task_q.task_done()
            self.result_q.put(result)


class MultiProcessor(object):
    """
    Convenience class for multiprocessing jobs.
    """

    def __init__(self, num_workers=0, verbose=True, progressbar=True):
        self._num_workers = num_workers
        self.verbose = verbose
        self.progressbar = progressbar
        if self._num_workers == 0:
            self._num_workers = multiprocessing.cpu_count()
        self._tasks = multiprocessing.JoinableQueue()
        self._results = multiprocessing.Queue()
        self._errors = multiprocessing.Queue()
        self._workers: List[Worker] = []
        self._num_tasks = 0
        self.total_time = 0

    def add_task(self, task):
        self._tasks.put(task)
        self._num_tasks += 1

    def close(self):
        self._results.close()
        self._errors.close()
        for w in self._workers:
            w.terminate()

    def run(self, read_results=True):
        # start N _workers
        start = timer()
        if self.verbose:
            print("Creating {:d} workers".format(self._num_workers))
        self._workers = [
            Worker(self._tasks, self._results, self._errors) for _ in range(self._num_workers)
        ]
        for w in self._workers:
            w.start()

        # add poison pills for _workers
        for i in range(self._num_workers):
            self._tasks.put(None)

        # write start message
        if self.verbose:
            print(
                "Running {:d} enqueued tasks and {:d} stop signals".format(
                    self._num_tasks, self._num_workers
                )
            )

        # check info on the queue, with a nice (somewhat stable) progressbar
        if self.progressbar:
            if self.verbose:
                print("waiting for the task queue to be filled...")
            num_wait = 0
            while self._tasks.empty():
                time.sleep(1)
                num_wait += 1
                if num_wait >= 5:
                    break
            tasks_now = self._num_tasks + self._num_workers
            pbar = tqdm.tqdm(total=tasks_now)
            while not self._tasks.empty():
                time.sleep(1)
                tasks_before = tasks_now
                tasks_now = self._tasks.qsize()
                resolved = tasks_before - tasks_now
                pbar.set_description("~{:7d} tasks remaining...".format(tasks_now))
                pbar.update(resolved)
            pbar.close()

        # join _tasks
        if self.verbose:
            print("waiting for all tasks to finish...")
        self._tasks.join()

        # check _errors
        if self.verbose:
            print("reading error queue... ")
        num_err = 0
        while not self._errors.empty():
            e, stacktrace = self._errors.get()
            num_err += 1
            print()
            print(stacktrace)
        if num_err >= 0:
            print("{} errors, check the log.".format(num_err))
        elif self.verbose:
            print("no errors found.")

        if not read_results:
            return self._results, self._num_tasks

        # read _results and return them
        if self.verbose:
            print("reading results...")
        results = []
        # # this can lead to some results missing
        # while not self._results.empty():
        while self._num_tasks > 0:
            result = self._results.get()
            results.append(result)
            self._num_tasks -= 1
        stop = timer()
        self.total_time = stop - start
        if self.verbose:
            print("Operation took {:.3f}s".format(self.total_time))
        return results


# ---------- Math ----------


def floor(num):
    return int(np.floor(num).astype(int))


def rnd(num):
    return int(np.round(num).astype(int))


def get_center_crop(h, w, th, tw):
    """
    Calculate centercrop (bigger image (h, w) target crop (th, tw)
    """

    crop_y = floor((h - th) / 2)
    crop_x = floor((w - tw) / 2)
    assert (
        crop_x >= 0 and crop_y >= 0 and crop_x + tw <= w and crop_y + th <= h
    ), "cropping {}x{} to {}x{}, crop top/left {}x{} bot/right {}x{}".format(
        w, h, tw, th, crop_x, crop_y, crop_x + tw, crop_y + th
    )
    return crop_y, crop_x, th, tw


def get_scaled_crop(h: int, w: int, target_h: int, target_w: int) -> Tuple[int, int, int, int]:
    """
    Calculate the position and size of a cropping rectangle, such that the input image will
    be cropped to the same aspect ratio as the target rectangle.

    After applying this crop, the output image can be scaled directly to the target rectangle
    without distortion (i.e. without changing its aspect ratio).

    Args:
        h: Input height
        w: Input width
        target_h: Target height
        target_w: Target width

    Returns:
        Crop position y, x, crop height, crop width
    """
    ratio_in = w / h
    ratio_out = target_w / target_h
    if ratio_in < ratio_out:
        # video too narrow
        crop_w = w
        crop_h = rnd(w / ratio_out)
    elif ratio_in > ratio_out:
        # video too wide
        crop_w = rnd(h * ratio_out)
        crop_h = h
    else:
        # video has correct ratio
        crop_w = w
        crop_h = h

    crop_x = floor((w - crop_w) / 2)
    crop_y = floor((h - crop_h) / 2)

    return crop_y, crop_x, crop_h, crop_w


def convert_actions_to_ffmpeg_filter(actions, verbose=False):
    if verbose:
        print("***** converting actions to ffmpeg filter")

    filter_str = ""
    for action_name, params in actions:
        if verbose:
            print("action {} params {} ".format(action_name, params), end="")

        if action_name == "resize":
            th, tw = params
            filter_str += "scale={:d}:{:d},".format(tw, th)
        elif action_name == "crop":
            cy, cx, ch, cw = params
            filter_str += "crop={:d}:{:d}:{:d}:{:d},".format(cw, ch, cx, cy)
        else:
            raise ValueError("action {} not recognized".format(action_name))
    if verbose:
        print("filter: {}".format(filter_str))
    return filter_str


def get_smart_crop_resize_actions(
    ih, iw, th, tw, crop_ratio=0.5, resize_min=1.1, resize_max=1.25, verbose=False
):
    """
    Calculates a smart crop from input width and height to target
    width and height, using the resize ratio (amount of resizing that would
    need to be done to go from input to target shape).

    - If the resize ratio is small enough (<=resize_min), resize directly
        (no cropping).
    - If the resize ratio is too big (>resize_max), crop the edges first until
        the resize ratio is at resize_max before doing the next step.
    - For the remaining images with resize_min < resize ratio <= resize_max,
        crop the borders (default 0.5 cuts 50% of the borders) and then resize.

    This function calculates the actions needed for this operation, which can
    then be used either directly on images or in ffmpeg.
    """
    input_ratio = iw / ih
    output_ratio = tw / th
    actions = []

    if verbose:
        print(
            "***** smart_crop from {}x{} ({:.2f}) to {}x{} ({:.2f})".format(
                ih, iw, input_ratio, th, tw, output_ratio
            )
        )

    # calculate amount of resizing done independent of sides
    resize_ratio = max(input_ratio / output_ratio, output_ratio / input_ratio)
    if resize_ratio < resize_min:
        if verbose:
            print("deviation {:.2f} is small enough: resize directly ".format(resize_ratio))
            actions.append(("resize", (th, tw)))
            return actions
    if verbose:
        print("resizing ratio is {:.2f}".format(resize_ratio))

    # some steps need to know whether image is too wide (crop x) or too
    # narrow (crop y)
    too_wide = True
    if input_ratio < output_ratio:
        too_wide = False

    newh, neww = ih, iw
    if resize_ratio > resize_max:
        # input is way out of bounds to crop smartly
        # centercrop to target_ratio to get image at resize_max ratio
        if too_wide:
            mh = ih
            target_ratio = output_ratio * resize_max
            mw = rnd(ih * target_ratio)
        else:
            mw = iw
            target_ratio = output_ratio / resize_max
            mh = rnd(mw / target_ratio)
        cy, cx, ch, cw = get_center_crop(ih, iw, mh, mw)
        actions.append(("crop", (cy, cx, ch, cw)))
        newh, neww = ch, cw

        if verbose:
            debugh, debugw = ch, cw
            middle_ratio = debugw / debugh
            deviation = max(middle_ratio / output_ratio, output_ratio / middle_ratio)
            print(
                "deviation {:.2f} too large: centercrop first. new image: "
                "{}x{} ({:.2f}) with deviation {:.2f}, cropped {}x{}".format(
                    resize_ratio, debugh, debugw, middle_ratio, deviation, ih - debugh, iw - debugw
                )
            )

    # now we want to crop crop_ratio of the edge
    # this is how many pixels we want to crop away
    if too_wide:
        # crop edge from width dimensions
        remove_edge = rnd(((neww - newh * output_ratio) / 2) * crop_ratio)
        neww -= remove_edge * 2
        actions.append(("crop", (0, remove_edge, newh, neww)))
        debug_removed_px = remove_edge / neww
    else:
        # crop edge from height dimensions
        remove_edge = rnd(((newh - neww / output_ratio) / 2) * crop_ratio)
        newh -= remove_edge * 2
        actions.append(("crop", (remove_edge, 0, newh, neww)))
        debug_removed_px = remove_edge / newh

    rh, rw = newh, neww
    if verbose:
        print(
            "removed {} pixels from the edge {:.2f}%, new image "
            "{}x{} ({:.2f})".format(remove_edge, debug_removed_px * 100, rh, rw, rw / rh)
        )

    # final resize
    actions.append(("resize", (th, tw)))

    return actions


# ---------- Video ----------


def get_ffprobe_streams(info):
    """
    :param info: info dictionary returned from ffprobe
    :return: video_stream, audio_stream
    """
    streams = info["streams"]
    assert len(streams) == 2, streams
    video_stream, audio_stream = None, None
    for s in streams:
        if s["codec_type"] == "video":
            video_stream = s
        elif s["codec_type"] == "audio":
            audio_stream = s
        else:
            raise ValueError("unknown stream: {}".format(s))
    assert video_stream["codec_type"] == "video", streams
    assert audio_stream["codec_type"] == "audio", streams
    if video_stream is None:
        raise ValueError("video stream not found")
    if audio_stream is None:
        raise ValueError("audio stream not found")
    return video_stream, audio_stream


def get_video_ffprobe_info(file_video: Union[str, Path]):
    """
    Return dictionary with info about the video given the input file
    Args:
        file_video:

    Returns:

    """
    # regular probe
    try:
        probe_info = ffmpeg.probe(str(file_video))
    except ffmpeg.Error as e:
        # ffmpeg custom errors can lead to problems in multiprocessing so just format them
        raise RuntimeError(f"ffprobe failed with {format_exception(e)} for file {file_video}\n"
                           f"stderr: {e.stderr}")
    # additional duration probe, otherwise duration is missing for some videos
    duration_call = (
        "ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 -i "
        f"{file_video}"
    )
    out, err, retcode = systemcall(duration_call)
    if retcode != 0:
        raise RuntimeError(
            f"Call {duration_call} failed with OUT={out} ERR={err} RETCODE={retcode}"
        )
    duration = float(out)
    probe_info["duration"] = duration
    return probe_info


def get_video_info_from_ffprobe_result(ffprobe_info):
    """
    :param ffprobe_info: dict with ffprobe _results
    :return: width, height, framerate as fraction string (e.g. 30/1)
    """
    video_stream, audio_stream = get_ffprobe_streams(ffprobe_info)
    height, width = video_stream["height"], video_stream["width"]
    fps = video_stream["r_frame_rate"]
    duration = ffprobe_info[
        "duration"
    ]  # this was set by running an additional ffprobe system command
    return width, height, fps, duration
