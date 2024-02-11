"""
Given a file with a list of question_ids, create a new subset of vqa val set.

"""
from copy import deepcopy
from pathlib import Path

from ovqa.paths import get_ovqa_cache_dir, get_ovqa_repo_root
from packg.iotools import load_json, dump_json


def create_vqav2_subsplit(ids_file, cache_dir=None, output_dir=None):
    if cache_dir is None:
        cache_dir = get_ovqa_cache_dir()
    if output_dir is None:
        output_dir = get_ovqa_repo_root() / "ovqa/annotations/vqav2"
    ids_file, cache_dir, output_dir = Path(ids_file), Path(cache_dir), Path(output_dir)
    name = ids_file.name.replace("_ids.json", "")
    meta_file = output_dir / f"{name}_meta.json"
    if meta_file.is_file():
        return

    print(f"Computing vqav2 split {name}...")
    new_ids = load_json(ids_file)
    new_ids_set = set(new_ids)
    print(f"Found {len(new_ids)} datapoints")

    vqav2_ann_cache_dir = cache_dir / "dataset_cache/vqav2/annotations"
    meta = load_json(vqav2_ann_cache_dir / "vqa_val_eval.json")
    answers = load_json(vqav2_ann_cache_dir / "v2_mscoco_val2014_annotations.json")
    questions = load_json(vqav2_ann_cache_dir / "v2_OpenEnded_mscoco_val2014_questions.json")
    meta_new = [a for a in meta if a["question_id"] in new_ids_set]

    answers_new = deepcopy(answers)
    answers_new["annotations"] = [
        a for a in answers["annotations"] if a["question_id"] in new_ids_set
    ]

    questions_new = deepcopy(questions)
    questions_new["questions"] = [
        a for a in questions["questions"] if a["question_id"] in new_ids_set
    ]
    dump_json(answers_new, output_dir / f"{name}_annotations.json")
    dump_json(questions_new, output_dir / f"{name}_questions.json")
    dump_json(meta_new, meta_file)
