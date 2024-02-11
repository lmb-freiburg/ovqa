import argparse
import logging
import numpy as np
import random
import torch
import torch.backends.cudnn as cudnn
from deepdiff import DeepDiff
from omegaconf import OmegaConf
from pathlib import Path
from pprint import pformat
from torch.utils.data import get_worker_info

import ovqa.tasks as tasks
from ovqa.common.lavis.config import Config
from ovqa.common.lavis.dist_utils import get_rank, init_distributed_mode, is_main_process
from ovqa.common.lavis.logger import setup_logger, add_file_handler_to_logger
from ovqa.common.lavis.utils import now
from ovqa.common.update_registry import update_registry
from ovqa.paths import (
    get_ovqa_output_dir,
    setup_ovqa_environ,
    print_all_environment_variables,
)
from ovqa.runners.runner_base import RunnerBase, setup_lavis_output_dir, get_lavis_output_dir
from ovqa.torchutils import count_params
from packg.debugging import connect_to_pycharm_debug_server


def parse_args(parser):
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    parser.add_argument("--trace", type=str, help="Connect debug server on this host.")
    parser.add_argument(
        "--trace_port", type=int, default=12345, help="Target debugging server port"
    )

    parser.add_argument("-c", "--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "-o",
        "--options",
        action="append",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument("-d", "--dataset-options", nargs="+", help="override dataset options")
    parser.add_argument("--debug", action="store_true", help="Modify config for debugging")
    parser.add_argument("--skip_existing", action="store_true", help="Skip existing output dir")
    parser.add_argument("--test_split", type=str, default=None, help="Override test split")
    parser.add_argument("--no_run", action="store_true", help="Do not run anything")
    parser.add_argument(
        "-a",
        "--add_config",
        action="append",
        help="Add this config to the configuration.",
    )
    args = parser.parse_args()
    return args


def main():
    parser = argparse.ArgumentParser(description="Evaluation")
    args = parse_args(parser)
    print(args)
    job_id = now()  # set before init_distributed_mode() to ensure same job_id across all ranks.
    if args.trace is not None:
        connect_to_pycharm_debug_server(args.trace, args.trace_port)

    setup_ovqa_environ()
    update_registry()

    cfg = Config(args, debug_mode=args.debug)
    init_distributed_mode(cfg.run_cfg)

    if is_main_process() and get_worker_info() is None:
        print_all_environment_variables()
    setup_seeds(cfg)
    setup_logger()  # set after init_distributed_mode() to only log on master.

    # update the output dir to add some more infos
    assert len(cfg.run_cfg.test_splits) == 1
    options_dict = cfg.return_opts_dict(args.options)
    if "suffix_output_dir" in options_dict.keys():
        suffix_output_dir = options_dict["suffix_output_dir"]
    else:
        suffix_output_dir = ""
    if "followup_prev_dir" in options_dict.keys():
        cfg.run_cfg.followup_prev_dir = options_dict["followup_prev_dir"]
        suffix_output_dir = (
            "~prev_"
            + options_dict["followup_prev_dir"].split("/")[-1].replace("~", "-")
            + suffix_output_dir
        )
    dataset_options = cfg.return_opts_dict(args.dataset_options)
    if "category_type" in dataset_options.keys():
        category_type = f"~{dataset_options['category_type']}"
    else:
        category_type = ""
    if "debug_max" in dataset_options.keys():
        debug_max = f"~{str(dataset_options['debug_max'])}"
    else:
        debug_max = ""

    # update output dir
    cfg.run_cfg.output_dir = (
        Path(cfg.run_cfg.output_dir)
        / f"{Path(args.cfg_path).parent.parts[-1]}{category_type}{debug_max}"
        / Path(args.cfg_path + suffix_output_dir).name.replace(".yaml", "", 1)
    ).as_posix()
    logging.info(f"Updated output dir: {cfg.run_cfg.output_dir}")

    # check if experiment already exists in lavis output_dir or $OVQA_OUTPUT_DIR
    output_dir = get_lavis_output_dir(cfg, job_id)
    p_output_dir = Path(output_dir)
    logging.info(f"Final output dir: {p_output_dir}")

    dirs_to_check = (
        p_output_dir,
        get_ovqa_output_dir() / p_output_dir.parent.name / p_output_dir.name,
    )
    logging.debug(f"Checking dirs: {dirs_to_check}")

    if args.skip_existing:
        for check_dir in dirs_to_check:
            logging.debug(f"Checking {check_dir}")
            files_to_check = []
            for files_to_glob in "evaluate.txt", "result/*":
                files_to_check += list(check_dir.glob(files_to_glob))
            if len(files_to_check) == 0:
                # no result files found so this folder is empty
                continue

            # check previous experiment has the same config as the new one
            old_config_file = check_dir / "config.yaml"
            new_config = OmegaConf.to_container(cfg.config, resolve=True)
            old_config = OmegaConf.to_container(OmegaConf.load(old_config_file), resolve=True)
            # compare configs
            ddiff = DeepDiff(old_config, new_config)
            values_changed = None
            if "values_changed" in ddiff.keys():
                values_changed = ddiff["values_changed"]
                values_changed = {
                    key: val
                    for key, val in values_changed.items()
                    if not any(
                        a in key
                        for a in [
                            "debug",
                            "batch_size",
                            "num_gpus",
                        ]
                    )
                }
            if values_changed is not None and len(values_changed) > 0:
                logging.warning("====================")
                logging.warning("Old experiment found with different config")
                logging.warning(f"{pformat(values_changed)}")
                logging.warning("====================")

            logging.warning("====================")
            logging.warning(f"Skip existing output dir {check_dir}")
            logging.warning("====================")
            return

    if args.no_run:
        logging.warning("====================")
        logging.warning(f"Skip running due to --no_run flag.")
        logging.warning("====================")
        return

    # add output dir to registry and log to file
    setup_lavis_output_dir(cfg, job_id)
    add_file_handler_to_logger(output_dir)

    cfg.pretty_print()
    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    logging.info(f"Build model")
    model = task.build_model(cfg)
    n_params = count_params(model)
    logging.info(f"Total params in model: {n_params / 10 ** 9:.3f}B")

    logging.warning(f"Start eval on rank {get_rank()}")

    runner = RunnerBase(cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets)
    runner.evaluate(skip_reload=True)


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


if __name__ == "__main__":
    main()
