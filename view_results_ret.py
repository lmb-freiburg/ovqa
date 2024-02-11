"""
Show retrieval results. This simply loads and combines the evaluate.txt files.
"""
from collections import defaultdict

import os
from attr import define
from loguru import logger
from pathlib import Path
from pprint import pprint
from typing import Optional

from ovqa.metrics.result_viewer_lib import sort_data_dict, visualize_data_dict
from ovqa.paths import get_ovqa_output_dir
from packg.iotools.jsonext import load_json
from packg.log import (
    SHORTEST_FORMAT,
    configure_logger,
    get_logger_level_from_args,
)
from typedparser import add_argument, TypedParser, VerboseQuietArgs


@define
class Args(VerboseQuietArgs):
    result_dir: Optional[Path] = add_argument(shortcut="-r", type=str, default=None)
    dataset_name: str = add_argument(shortcut="-d", type=str, default="imagenet1k")
    dataset_split: str = add_argument(shortcut="-s", type=str, default="val")
    width_terminal: Optional[int] = add_argument(
        shortcut="-t", type=int, default=None, help="Overwrite width of terminal"
    )
    no_browser: bool = add_argument(shortcut="-n", action="store_true", help="Do not open browser")


def main():
    parser = TypedParser.create_parser(Args, description=__doc__)
    args: Args = parser.parse_args()
    configure_logger(level=get_logger_level_from_args(args), format=SHORTEST_FORMAT)
    logger.info(f"{args}")

    # load results
    result_dir = args.result_dir
    if result_dir is None:
        result_dir = get_ovqa_output_dir() / f"retrieval~{args.dataset_name}~{args.dataset_split}"
    logger.info(f"Loading results from {result_dir}")
    result_names = sorted(os.listdir(result_dir))
    data_dict = defaultdict(dict)
    for result_name in result_names:
        single_result_dir = result_dir / result_name
        if not single_result_dir.is_dir():
            continue
        evaluate_txt = single_result_dir / "evaluate.txt"
        if not evaluate_txt.is_file():
            logger.info(f"{evaluate_txt} does not exist, skipping dir")
            continue
        evaluate_all_data = load_json(evaluate_txt)
        assert (
            args.dataset_split in evaluate_all_data
        ), f"Missing {args.dataset_split} in {evaluate_txt}: {evaluate_all_data}"
        evaluate_data = evaluate_all_data[args.dataset_split]

        del evaluate_data["agg_metrics"]
        for metric_name, metric_value in evaluate_data.items():
            data_dict[result_name][metric_name] = metric_value

    # sort from dict modelname -> metrics -> values
    # to dict columnname -> list of column data
    all_columns = set(key for vals in data_dict.values() for key in vals.keys())
    data_dict_new = defaultdict(list)
    for model_name, data_dict_values in data_dict.items():
        data_dict_new["model"].append(model_name)
        for column_name in all_columns:
            if column_name not in data_dict_values:
                data_dict_new[column_name].append(-1)
            else:
                data_dict_new[column_name].append(data_dict_values[column_name])
    data_dict = data_dict_new
    pprint(data_dict)

    # sort output columns
    sort_order = (
        "model",
        "acc1",
        "acc5",
    )
    new_data_dict = sort_data_dict(data_dict, sort_order)
    pprint(new_data_dict)

    format_fns = {k: "{:.2%}" for k in sort_order[1:]}
    # display in console, save as html and as csv
    visualize_data_dict(
        new_data_dict,
        format_fns,
        result_dir,
        args.dataset_name,
        args.dataset_split,
        open_browser=not args.no_browser,
        width_terminal=args.width_terminal,
        prefix="retrieval~"
    )


if __name__ == "__main__":
    main()
