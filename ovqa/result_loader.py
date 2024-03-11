import argparse
from attr import define
from loguru import logger
from pathlib import Path
from typing import Dict, Any, Optional, List

from packg.constclass import Const
from packg.iotools import make_git_pathspec
from packg.iotools.jsonext import load_json
from packg.typext import PathType


@define
class ResultInterface:
    path: Path
    split: str

    @classmethod
    def from_path(cls, path: Path, split: str = "val"):
        return cls(path, split)

    def load_output(self, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError


@define
class ResultBlip2Qa(ResultInterface):
    """
    Single generated answer per question (no probabilities etc)

    Files expected in the path:
        evaluate.txt - json file e.g.
            {"agg_metrics": 30.89, "other": 8.85, "yes/no": 63.38, "number": 20.63}

        result/val_ebochbest.json
            [{"question_id": 1, "caption": "two trains"}, ...]
    """

    def get_result_file(self):
        vqa_result_path = Path(self.path) / f"result/{self.split}_vqa_result.json"
        return vqa_result_path

    def load_output(self, num2key: Optional[Dict[int, str]] = None) -> Dict[str, str]:
        """
        Load the output from the result directory
        Args:
            num2key: Lavis output has keys '0', '1', '2', ... if this dict is given,
                keys are converted

        Returns:

        """
        # vqa_result = load_json(Path(self.path) / f"result/{self.split}_vqa_result.json")
        vqa_result = load_json(self.get_result_file())
        if "answer" in vqa_result[0].keys():
            vqa_result_dict = {item["question_id"]: item["answer"] for item in vqa_result}
        elif "answers" in vqa_result[0].keys():
            vqa_result_dict = {item["question_id"]: item["answers"] for item in vqa_result}
        elif "pred_ans" in vqa_result[0].keys():
            # gqa format
            vqa_result_dict = {item["question_id"]: item["pred_ans"] for item in vqa_result}
        else:
            raise ValueError(f"Unknown result format {vqa_result[0].keys()}")
        if num2key is not None:
            vqa_result_dict = {num2key[int(k)]: v for k, v in vqa_result_dict.items()}
        return {str(k): v for k, v in vqa_result_dict.items()}

    def load_followup_info(self):
        """

        Returns:
            followup info like:
            {
                "val_00000007": {
                    "status": "correct"
                },
                ...
                "val_00049051": {
                    "status": "failed"
                },
                ...
                "val_00049998": {
                    "status": "followup",
                    "object": "dog"
                },
            }



        """
        return load_json(Path(self.path) / "result/followup.json")


@define
class ResultBlip2Cap(ResultInterface):
    """
    Single generated caption per image

    Files expected in the path:
        evaluate.txt - json file e.g.
            {"agg_metrics": 30.89, "other": 8.85, "yes/no": 63.38, "number": 20.63}

        result/val_epochbest.json
            [{"image_id": 1, "caption": "two trains"}, ...]
    """

    def get_result_file(self):
        vqa_result_path = Path(self.path) / f"result/{self.split}_epochbest.json"
        return vqa_result_path

    def load_output(self, num2key: Optional[Dict[int, str]] = None) -> Dict[str, str]:
        # vqa_result = load_json(Path(self.path) / f"result/{self.split}_epochbest.json")
        vqa_result = load_json(self.get_result_file())
        vqa_result_dict = {item["image_id"]: item["caption"] for item in vqa_result}

        if num2key is not None:
            vqa_result_dict = {num2key[int(k)]: v for k, v in vqa_result_dict.items()}
        return {str(k): v for k, v in vqa_result_dict.items()}


class ResultsFormatConst(Const):
    BLIP2_QA = "blip2_qa"
    BLIP2_CAP = "blip2_cap"


def read_results_dir(
    result_dir: PathType,
    result_format="auto",
    include_list: Optional[List[str]] = None,
    split: str = "val",
):
    """
    Read all results in a directory
    """
    if include_list is not None:
        logger.info(f"Using include list: {include_list}")
    results_dict = {}
    for pth in sorted(Path(result_dir).glob("*")):
        load_result = True
        if not pth.is_dir():
            continue
        logger.debug(f"Check {pth} for results")
        if include_list is not None:
            load_result = False
            for include_item in include_list:
                if include_item in pth.name:
                    load_result = True
                    break
        if load_result:
            result_obj = read_single_result(pth, result_format=result_format, split=split)
            if result_obj is None:
                logger.debug(f"Skipping {pth} - no result found")
                continue

            results_dict[pth.name] = result_obj
    return results_dict


def read_single_result(
    pth: PathType,
    result_format: str = "auto",
    split: str = "val",
    ignore_errors: bool = True,
) -> Optional[ResultInterface]:
    """
    Read a single result
    """
    if result_format == "auto":
        result_format_here = detect_result_format(pth, split=split, ignore_errors=ignore_errors)
    else:
        result_format_here = result_format

    if result_format_here is None:
        return None

    if result_format_here == ResultsFormatConst.BLIP2_QA:
        result_obj = ResultBlip2Qa.from_path(pth, split=split)
    elif result_format_here == ResultsFormatConst.BLIP2_CAP:
        result_obj = ResultBlip2Cap.from_path(pth, split=split)
    else:
        raise ValueError(f"Unknown result format {result_format}")
    return result_obj


def detect_result_format(
    pth: PathType, split: str = "val", ignore_errors: bool = True
) -> Optional[str]:
    pth = Path(pth)
    if (pth / "result" / f"{split}_vqa_result.json").is_file():
        result_format_here = ResultsFormatConst.BLIP2_QA
    elif (pth / "result" / f"{split}_epochbest.json").is_file():
        result_format_here = ResultsFormatConst.BLIP2_CAP
    else:
        errmsg = f"Could not determine result format for {pth} with split {split}"
        if ignore_errors:
            logger.warning(errmsg)
        else:
            raise ValueError(errmsg)

        result_format_here = None
    return result_format_here


def main():
    # quick testing of results
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-r",
        "--result_dir",
        type=str,
        default="/misc/lmbssd/gings/results/clean/imagenet1k-square~val/blip2-t5xl~pt~vqa~qa-short1~followup",
    )
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    result_obj = read_single_result(result_dir)
    print(result_obj.load_output())
    print(result_obj.load_followup_output())


if __name__ == "__main__":
    main()
