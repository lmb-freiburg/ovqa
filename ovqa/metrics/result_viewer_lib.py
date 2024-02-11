"""
Components reused by the view_results_* scripts
"""

from collections import defaultdict

import hashlib
import numpy as np
import pandas as pd
import pickle
from attrs import define
from loguru import logger
from pathlib import Path
from typing import Optional, List

from packg.dtime import get_timestamp_for_filename
from packg.strings import b64_encode_from_bytes
from typedparser import add_argument, VerboseQuietArgs
from visiontext.pandatools import save_df_to_html, display_df


@define
class ResultViewerArgs(VerboseQuietArgs):
    result_dir: Optional[Path] = add_argument(shortcut="-r", type=str, default=None)
    dataset_name: str = add_argument(shortcut="-d", type=str, default=None, required=True)
    dataset_split: str = add_argument(shortcut="-s", type=str, default=None, required=True)
    trace: Optional[str] = add_argument(type=str, default=None, help="Target debugging server host")
    trace_port: int = add_argument(type=int, default=12345, help="Target debugging server port")
    width_terminal: Optional[int] = add_argument(
        shortcut="-t", type=int, default=None, help="Overwrite width of terminal"
    )
    no_cache: bool = add_argument(action="store_true", help="Disable cache")
    no_browser: bool = add_argument(shortcut="-n", action="store_true", help="Do not open browser")
    split_model_names: bool = add_argument(
        shortcut="-m", action="store_true", help="Split model names at ~ into columns"
    )
    include_models: Optional[List[str]] = add_argument(
        shortcut="-i",
        type=str,
        help="list of models to evaluate",
        action="append",
    )
    include_models_file: Optional[Path] = add_argument(
        type=str, help="file with model names to evaluate, one model per line"
    )
    llm: bool = add_argument(action="store_true", help="Use LLM metrics")
    dont_cut_long_text: bool = add_argument(
        shortcut="-c",
        action="store_true",
        help="Disable cutting long sentences and removing 'long answer: ' etc.",
    )
    options: Optional[list] = add_argument(
        shortcut="-o",
        action="append",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    std_mode: Optional[str] = add_argument(
        shortcut="-a",
        type=str,
        help="Select a mode to aggregate results and calculate mean/std",
    )
    std_ddof: int = add_argument(
        type=int,
        default=1,
        help="Delta degrees of freedom. The divisor used in calculations "
        "is (N - ddof), where N represents the number of elements.",
    )


def hash_predictions(t_pred_values):
    hasher = hashlib.sha3_224()
    for pred in t_pred_values:
        hasher.update(pred.encode())
    out_hash_bytes = hasher.digest()
    out_hash = b64_encode_from_bytes(out_hash_bytes, strip_equals=True)
    return out_hash


def load_metric_cache(metric_file, readable_time, pred_hash):
    if not metric_file.is_file():
        return None
    metric_data = pickle.load(open(metric_file, "rb"))
    if readable_time != metric_data["readable_time"]:
        logger.info(
            f"Time changed for {metric_file}: " f"{readable_time} != {metric_data['readable_time']}"
        )
        return None
    if pred_hash != metric_data["pred_hash"]:
        logger.info(
            f"Pred hash changed for {metric_file}: " f"{pred_hash} != {metric_data['pred_hash']}"
        )
        return None
    return metric_data["scores"]


def write_metric_cache(scores, metric_file, readable_time, pred_hash):
    metric_data = {
        "scores": scores,
        "readable_time": readable_time,
        "pred_hash": pred_hash,
    }
    pickle.dump(metric_data, open(metric_file, "wb"))


def return_opts_dict(dotlist):
    dotdict = {}
    if not isinstance(dotlist, (list, tuple)):
        return dotdict
    for arg in dotlist:
        if not isinstance(arg, str):
            continue
        idx = arg.find("=")
        if idx == -1:
            key = arg
            value = None
        else:
            key = arg[0:idx]
            value = arg[idx + 1 :]

        dotdict[key] = value
    return dotdict


def sort_data_dict(data_dict, sort_order, split_model_names=False, std_mode=None, std_ddof=1):
    remaining_keys = set(data_dict.keys())
    model_info_dict, metric_value_dict = {}, {}

    if std_mode is not None:
        # must split the model to find out what is needed to use std_mode
        split_model_names = True

    split_fields = ["model", "data", "hyperp", "prompt", "question", "additional"]
    for key in sort_order:
        if key not in data_dict:
            continue
        if key == "models" and split_model_names:
            model_names = data_dict[key]
            model_names_split = [name.split("~") for name in model_names]
            model_names_split_max_len = max(len(name) for name in model_names_split)
            for i in range(model_names_split_max_len):
                new_column = []
                for name_split in model_names_split:
                    if len(name_split) > i:
                        new_column.append(name_split[i])
                    else:
                        new_column.append("")
                if i >= len(split_fields):
                    col_name = f"unknown{i-len(split_fields)}"
                else:
                    col_name = split_fields[i]
                model_info_dict[col_name] = new_column
            remaining_keys.remove(key)
            continue

        metric_value_dict[key] = data_dict[key]
        remaining_keys.remove(key)
    for key in sorted(remaining_keys):
        metric_value_dict[key] = data_dict[key]

    if std_mode is not None:
        new_data_dict = apply_std_mode(model_info_dict, metric_value_dict, std_mode, std_ddof)
    else:
        new_data_dict = {**model_info_dict, **metric_value_dict}

    return new_data_dict


def format_pandas_df(df, format_functions, split_at: Optional[str] = "."):
    """

    Args:
        df: data frame
        format_functions: dict of column name ->
            either callable formatting functions or objects with a .format method
        split_at: str, split column name at this char and use the first part as metric name

    Returns:

    """
    format_fns = {}
    for column in df.columns:
        metric_name = column.split(split_at)[0] if split_at is not None else column
        if metric_name in format_functions:
            fn = format_functions[metric_name]
            if hasattr(fn, "format"):
                fn = fn.format
            format_fns[column] = fn
    df_formatted = df.copy()
    for column, format_fn in format_fns.items():
        df_formatted[column] = df_formatted[column].apply(format_fn)
    return df_formatted


STD_TEXT = "_std"


def visualize_data_dict(
    new_data_dict,
    all_metrics,
    result_dir,
    dataset_name=None,
    dataset_split=None,
    open_browser=True,
    width_terminal=None,
    prefix="",
):
    df = pd.DataFrame(new_data_dict)

    format_functions = all_metrics  # metrics have .format so they can be used as formatters
    # duplicate the formatting for std columns
    for k, v in list(format_functions.items()):
        format_functions[f"{k}{STD_TEXT}"] = v

    df_formatted = format_pandas_df(df, all_metrics)
    display_df(df_formatted, width=width_terminal)

    random_name = f"{get_timestamp_for_filename()}"
    html_file = Path(
        f"{result_dir}/../tmphtmloutput/{prefix}{dataset_name}/{random_name}.html"
    ).resolve()

    save_df_to_html(
        df_formatted,
        f"{prefix}{dataset_name} {dataset_split} results from directory {Path(result_dir).as_posix()}.",
        html_file,
        open_browser=open_browser,
    )
    logger.info(f"Saved HTML output to {html_file}")

    # save csv
    csv_file = Path(
        f"{result_dir}/../tmphtmloutput/{prefix}{dataset_name}/{random_name}.csv"
    ).resolve()
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_file)
    logger.info(f"Saved CSV as {csv_file}")
    logger.info(f"Data source was {result_dir}")


def apply_std_mode(model_info_dict, metric_value_dict, std_mode, std_ddof: int):
    """

    Args:
        model_info_dict: {column_name -> list of values}
            keys: model data hyperp prompt question additional
            values are str
        metric_value_dict: {column_name -> list of values}
            N ega1 ega5 ega_synmax1 ega_synmax5 cont% ...
            values are numbers
        std_mode:
            for imagenet aggregate over the question
        std_ddof: delta degrees of freedom. the divisor used in calculations
            is (N - ddof), where N represents the number of elements.

    Returns:

    """
    n_rows = len(next(iter(model_info_dict.values())))

    # these keys should be removed and aggregated mean/std over
    aggregate_keys = None
    if std_mode == "question":
        aggregate_keys = ["question"]
    assert aggregate_keys is not None, f"Unknown std_mode {std_mode}"

    # group all row numbers by the remaining columns
    remaining_keys = [m for m in model_info_dict.keys() if m not in aggregate_keys]
    aggregated = defaultdict(list)
    for n in range(n_rows):
        model_info_values = tuple(model_info_dict[k][n] for k in remaining_keys)
        aggregated[model_info_values].append(n)

    # build the new aggregated info dict
    out_dict = defaultdict(list)
    for key_tuple, row_numbers in aggregated.items():
        n_values = len(row_numbers)

        # add the remaining, non-aggregated keys to the table
        for i, key in enumerate(remaining_keys):
            out_dict[key].append(key_tuple[i])
        out_dict["std_n"].append(n_values)

        # aggregate metrics over the found row numbers and add them to the table
        for col, values in metric_value_dict.items():
            values_here = [values[i] for i in row_numbers]
            values_mean = np.mean(values_here)
            if len(values_here) == 1:
                logger.error(
                    f"Computing std for only one experiment will produce NaN. "
                    f"Key {key_tuple} column {col}"
                )

            values_std = np.std(values_here, ddof=std_ddof)
            out_dict[f"{col}"].append(values_mean)
            out_dict[f"{col}{STD_TEXT}"].append(values_std)
    return out_dict


def ensure_compatible_keys(targets: dict, pred_keys: list):
    """correct int and str missmatch type"""
    # print("target", list(targets.keys())[0], type(list(targets.keys())[0]))
    # print("prediction", list(pred_keys)[0], type(list(pred_keys)[0]))
    if type(list(targets.keys())[0]) != type(list(pred_keys)[0]):  # noqa
        # convert everything to str
        if isinstance(list(targets.keys())[0], int):
            new_targets = {str(key): val for key, val in targets.items()}
            targets = new_targets
        elif isinstance(list(pred_keys)[0], int):
            new_pred_keys = [str(key) for key in pred_keys]
            pred_keys = new_pred_keys
    assert type(list(targets.keys())[0]) == type(list(pred_keys)[0]), (  # noqa
        f"Format type of predicted ({type(list(pred_keys)[0])}) and "
        f"target ({type(list(targets.keys())[0])}) should be the same."
    )
    return targets, pred_keys
