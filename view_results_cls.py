"""
Show oVQA classification results.
"""

from collections import defaultdict
from pathlib import Path

import os
import time
import torch
from loguru import logger
from tqdm import tqdm
from typing import Dict

from ovqa.datasets.interface_metadata import ClsMetadataInterface
from ovqa.datasets.meta_loading import meta_loader
from ovqa.metrics.load_metrics import setup_clipmatch_metrics, setup_text_metrics
from ovqa.metrics.result_viewer_lib import (
    hash_predictions,
    sort_data_dict,
    visualize_data_dict,
    ResultViewerArgs,
    ensure_compatible_keys,
    load_metric_cache,
    write_metric_cache,
)
from ovqa.metrics.torchmetrics_ext import MetricCollectionExt
from ovqa.paths import get_ovqa_output_dir
from ovqa.result_loader import read_results_dir
from ovqa.textutils.cut_sentence import cut_too_long_text
from packg.debugging import connect_to_pycharm_debug_server
from packg.log import (
    configure_logger,
    get_logger_level_from_args,
    SHORTER_FORMAT,
)
from ovqa.paths import print_all_environment_variables
from typedparser import TypedParser


def main():
    parser = TypedParser.create_parser(ResultViewerArgs, description=__doc__)
    args: ResultViewerArgs = parser.parse_args()
    configure_logger(level=get_logger_level_from_args(args), format=SHORTER_FORMAT)
    logger.info(f"Args: {args}")
    print_all_environment_variables(logger.info)
    if args.trace is not None:
        connect_to_pycharm_debug_server(args.trace, args.trace_port)

    # load results
    result_dir = args.result_dir
    if result_dir is None:
        result_dir = get_ovqa_output_dir() / f"{args.dataset_name}~{args.dataset_split}"
    assert result_dir.is_dir(), f"Result dir {result_dir} does not exist"
    logger.info(f"Loading results from {result_dir}")
    include_list = args.include_models
    if include_list is None and args.include_models_file is not None:
        logger.info(f"Reading model list from {args.include_models_file}")
        include_list = args.include_models_file.read_text(encoding="utf-8").splitlines()
    results = read_results_dir(result_dir, include_list=include_list, split=args.dataset_split)
    logger.info(f"Checking results: {sorted(results.keys())}")

    # load classification dataset
    meta: ClsMetadataInterface = meta_loader.load_metadata(args.dataset_name, args.dataset_split)
    class_names = meta.get_class_list()
    targets = meta.get_targets()
    logger.debug(f"Class names: {class_names}")
    logger.info(f"Got {len(class_names)} classes")

    # load metrics
    clipmatch_metrics: MetricCollectionExt = setup_clipmatch_metrics(
        class_names, meta.templates_name
    )
    text_metrics: MetricCollectionExt = setup_text_metrics()
    all_metrics = {**clipmatch_metrics, **text_metrics}
    logger.info(f"Metrics to use: {list(all_metrics.keys())}")

    # loop results
    data_dict = defaultdict(list)
    pbar = tqdm(total=len(results), desc="Processing CLS results")
    for result_name, result in results.items():
        logger.debug(f"Processing {result_name}")
        data_dict["models"].append(result_name)
        preds: Dict[str, str] = result.load_output(num2key=meta.get_num2key())
        pred_keys = list(preds.keys())
        pred_values = list(preds.values())
        if not args.dont_cut_long_text:
            new_pred_values = []
            for pred in pred_values:
                new_pred_value = cut_too_long_text(pred)
                if pred.strip() != new_pred_value:
                    logger.debug(f"==========" f"\n    {pred}\n    {new_pred_value}\n")
                new_pred_values.append(new_pred_value)
            pred_values = new_pred_values

        data_dict["N"].append(len(pred_values))
        targets, pred_keys = ensure_compatible_keys(targets, pred_keys)
        r_file = result_dir / f"{result_name}/result/{args.dataset_split}_vqa_result.json"

        # get mod date of result file and hash of predictions. recompute metrics in case of changes
        mod_time = os.stat(r_file).st_mtime
        readable_time = time.ctime(mod_time)
        logger.debug(f"Last modification time: {readable_time}")
        hashed_preds = hash_predictions(pred_values)

        # compute clipmatch metrics
        target_ids = [targets[key] for key in pred_keys]
        target_text = [class_names[class_idx] for class_idx in target_ids]
        results_dict_scores, results_dict_mean = {}, {}
        for metric_name, metric in clipmatch_metrics.items():
            metric_file = Path(r_file).parent / f"{args.dataset_split}_vqa_scores_{metric_name}.pkl"
            scores = load_metric_cache(metric_file, readable_time, hashed_preds)
            if scores is None:
                logger.debug(f"Recomputing metric {metric_name} for {result_name}")
                metric.reset()
                metric.update(pred_keys, pred_values, target_ids)
                scores = metric.compute_per_datapoint(return_dict=True)["scores"]
                write_metric_cache(scores, metric_file, readable_time, hashed_preds)
            else:
                logger.debug(f"Metric {metric_name} for {result_name} reloaded from cache")
            results_dict_scores[metric_name] = scores
            results_dict_mean[metric_name] = torch.mean(scores).item()

        # compute text metrics
        for metric_name, metric in text_metrics.items():
            metric_file = Path(r_file).parent / f"{args.dataset_split}_vqa_scores_{metric_name}.pkl"
            scores = load_metric_cache(metric_file, readable_time, hashed_preds)
            if scores is None:
                logger.debug(f"Recomputing metric {metric_name} for {result_name}")
                metric.reset()
                metric.update(pred_values, target_text)
                scores = metric.compute_per_datapoint(return_dict=True)["scores"]
                write_metric_cache(scores, metric_file, readable_time, hashed_preds)
            else:
                logger.debug(f"Metric {metric_name} for {result_name} reloaded from {metric_file}")

            results_dict_scores[metric_name] = scores
            results_dict_mean[metric_name] = torch.mean(scores).item()

        for score_name, score_value in results_dict_mean.items():
            data_dict[score_name].append(score_value)

        time.sleep(0.01)
        pbar.update()
    pbar.close()
    print(data_dict.keys())

    # sort output columns
    new_data_dict = sort_data_dict(
        data_dict,
        (
            "models",
            "N",
            *list(all_metrics.keys()),
        ),
        split_model_names=args.split_model_names,
        std_mode=args.std_mode,
        std_ddof=args.std_ddof,
    )

    # display in console, save as html and as csv
    visualize_data_dict(
        new_data_dict,
        all_metrics,
        result_dir,
        args.dataset_name,
        args.dataset_split,
        open_browser=not args.no_browser,
        width_terminal=args.width_terminal,
    )


if __name__ == "__main__":
    main()
