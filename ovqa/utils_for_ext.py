import logging
import os
from typing import Dict, List, Optional

from packg.iotools.jsonext import load_json


def barrier_safe_ext():
    """Barrier only if in a distributed torch run. Does not fail if torch package is missing."""
    if ("RANK" in os.environ or "LOCAL_RANK" in os.environ) and "WORLD_SIZE" in os.environ:
        from torch import distributed as dist

        dist.barrier()


def make_list_smaller(input_list, start_num=0, max_amount=None):
    input_list = input_list[start_num:]
    if max_amount is not None:
        input_list = input_list[:max_amount]
    return input_list


def load_captions_maybe(config, instance_ids=None) -> Optional[Dict[int, List[str]]]:
    # load captions into dataset
    question_caption_file = config.get("question_caption_file", None)
    if question_caption_file is None:
        return None
    # question_caption_file = autoupdate_path(question_caption_file)
    captions = load_json(question_caption_file)
    captions_dict = {caption["question_id"]: caption["caption"] for caption in captions}
    if instance_ids is not None:
        question_captions = {qid: captions_dict[qid] for qid in instance_ids}
        logging.info(f"Loaded captions for {len(instance_ids)} ids from {question_caption_file}")
    else:
        question_captions = captions_dict
        logging.info(f"Loaded all captions, {len(captions_dict)} ids from {question_caption_file}")
    return question_captions
