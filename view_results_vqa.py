"""
Show classical VQA results.
"""
from collections import defaultdict
from pathlib import Path

import io
import os
import time
import torch
from loguru import logger
from typing import Dict

from ovqa import tasks
from ovqa.common.lavis.config import Config
from ovqa.datasets.lavis.vqav2_datasets import VQAv2EvalDataset
from ovqa.metrics.result_viewer_lib import (
    ResultViewerArgs,
    hash_predictions,
    sort_data_dict,
    visualize_data_dict,
    load_metric_cache,
    write_metric_cache,
)
from ovqa.metrics.vqa_loader import get_vqav2_metrics, get_singleanswer_vqa_metrics
from ovqa.paths import get_ovqa_output_dir
from ovqa.result_loader import read_results_dir
from ovqa.tasks import VQATask
from ovqa.textutils.cut_sentence import cut_too_long_text
from packg.debugging import connect_to_pycharm_debug_server
from packg.iotools import dumps_yaml
from packg.log import configure_logger, get_logger_level_from_args, SHORTER_FORMAT
from packg.paths import print_all_environment_variables
from packg.tqdmext import tqdm_max_ncols
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
    logger.info(f"Found {len(results)} results:\n" + "\n".join(results.keys()) + "\n")

    # load vqa dataset, build pseudo config to create datasets.
    # Note: model and most of the run config are never used, but needed to create the task
    config = {
        "datasets": {args.dataset_name: {"type": "eval"}},
        "run": {
            "task": "vqa",
            "test_splits": [args.dataset_split],
            "evaluate": True,
            "distributed": False,
            "num_beams": None,
            "max_new_tokens": None,
            "min_new_tokens": None,
            "prompt": None,
            "inference_method": None,
        },
        "model": {
            "arch": "blip2_t5",
            "model_type": "pretrain_flant5xl",
        },
    }
    config_stream = io.StringIO(dumps_yaml(config))
    cfg = Config.from_file(config_stream)
    task: VQATask = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    dataset: VQAv2EvalDataset = datasets[args.dataset_name][args.dataset_split]
    answers = {str(ann["question_id"]): ann["answer"] for ann in dataset.annotation}
    questions = {str(ann["question_id"]): ann["question"] for ann in dataset.annotation}

    # load metrics depending on the dataset
    if args.dataset_name == "vqav2":
        multi_answer = True
        metrics = get_vqav2_metrics(dataset.answer_list, llm=args.llm)
    elif args.dataset_name == "gqa":
        multi_answer = False
        metrics = get_singleanswer_vqa_metrics(llm=args.llm)
    else:
        raise ValueError(f"Unknown dataset {args.dataset_name}")
    all_metrics = {**metrics}
    logger.info(f"Metrics to use: {list(all_metrics.keys())}")

    # loop results
    data_dict = defaultdict(list)
    pbar = tqdm_max_ncols(total=len(results), desc="Processing VQA results")
    for result_name, result in results.items():
        logger.debug(f"Processing {result_name}")
        data_dict["models"].append(result_name)
        preds: Dict[str, str] = result.load_output()  # todo num2key=meta.get_num2key())
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

        answers_here = [answers[str(key)] for key in pred_keys]
        questions_here = [questions[str(key)] for key in pred_keys]
        data_dict["N"].append(len(pred_values))
        r_file = result_dir / f"{result_name}/result/{args.dataset_split}_vqa_result.json"

        # get mod date of result file and hash of predictions. recompute metrics in case of changes
        mod_time = os.stat(r_file).st_mtime
        readable_time = time.ctime(mod_time)
        logger.debug(f"Last modification time: {readable_time}")
        hashed_preds = hash_predictions(pred_values)

        # compute vqa metrics
        results_dict_scores, results_dict_mean = {}, {}
        for metric_name, metric in all_metrics.items():
            metric_file = Path(r_file).parent / f"{args.dataset_split}_vqa_scores_{metric_name}.pkl"
            scores = load_metric_cache(metric_file, readable_time, hashed_preds)
            if scores is None:
                metric.reset()
                metric.update(pred_values, answers_here, questions_here)
                if multi_answer:
                    per_answer_list_of_list = metric.compute_per_answer()
                    # per_answr = {key: val for key, val in zip(pred_keys, per_answer_list_of_list)}
                    scores = torch.tensor(metric.aggregate_vqa_scores(per_answer_list_of_list))
                else:
                    scores = metric.compute_per_datapoint()
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
