"""
Import and run inference on all models used in the oVQA paper for testing.

python check_all_models.py -r
"""

import argparse
import gc
import logging
from pprint import pprint

import torch
from PIL import Image
from attr import define

from ovqa.common.lavis.logger import setup_logger
from ovqa.models import model_zoo, load_model_and_preprocess
from ovqa.paths import get_ovqa_repo_root
from ovqa.tasks import MultimodalClassificationTask
from typedparser import VerboseQuietArgs, add_argument, TypedParser
from visiontext.mathutils import torch_stable_softmax


@define
class Args(VerboseQuietArgs):
    all_models: bool = add_argument(shortcut="-a", action="store_true", help="Check all models")
    run_inference: bool = add_argument(shortcut="-r", action="store_true", help="Run VQA example")
    model_name: str | None = add_argument(shortcut="-m", default=None)


@torch.inference_mode()
def main():
    parser = TypedParser.create_parser(Args, description=__doc__)
    args: Args = parser.parse_args()
    setup_logger()
    all_combinations = []
    print(f"Models in model_zoo:\n{model_zoo}\n")
    for model_name, model_types in model_zoo:
        for model_type in model_types:
            all_combinations.append((model_name, model_type))
    vqa_models = [
        ("blip_vqa", "vqav2"),
        ("x2vlm", "x2vlm_base1b_ftvqa"),
        ("x2vlm", "x2vlm_large1b_ftvqa"),
        ("blip2_opt", "pretrain_opt2.7b"),
        ("blip2_t5", "pretrain_flant5xl"),
        ("blip2_t5_instruct", "flant5xl"),
        ("blip2_vicuna_instruct", "vicuna7b"),
        ("llava", "llava7b"),
    ]
    retrieval_models = [
        ("blip2", "coco"),
        ("blip2", "pretrain"),
        ("clip", "ViT-L-14"),
        ("openclip", "default"),  # default is defined as EVA01-g-14/laion400m_s11b_b41k
    ]
    check_models = vqa_models + retrieval_models
    if args.all_models:
        check_models = all_combinations

    if args.model_name is not None:
        for model_name, model_type in all_combinations:
            if args.model_name == model_name:
                check_models = [(model_name, model_type)]
                print(f"Wil check model {args.model_name=}")
                break
        else:
            raise ValueError(f"{args.model_name=} not found in {[k for k, v in all_combinations]}")

    example_image = Image.open(get_ovqa_repo_root() / "assets/example.png").convert("RGB")
    print(f"Got example input image: {example_image}")

    print(f"Models to check:")
    pprint(check_models)
    for model_i, (model_name, model_type) in enumerate(check_models):
        logging.info(
            f"********** Checking {model_i:3d}/{len(check_models)}: " f"{model_name} / {model_type}"
        )
        model, vis_processors, txt_processors = load_model_and_preprocess(
            model_name, model_type, is_eval=False
        )

        if not args.run_inference:
            continue

        vis_proc_eval = vis_processors["eval"]
        txt_proc_eval = txt_processors["eval"]
        image_tensor = vis_proc_eval(example_image).unsqueeze(0).cuda()

        if (model_name, model_type) in vqa_models:
            print(f"Run VQA inference for {model_name} / {model_type}")
            model = model.cuda()
            question = "What is this?"
            question_proc = txt_proc_eval(question)
            print(f"Input image {image_tensor.shape} and question: {question_proc}")
            samples = {
                "image": image_tensor,
                "text_input": [question_proc],
            }
            answers = model.predict_answers(samples, prompt="{}")
            print(f"Model output: {answers[0]}")
            del model, vis_processors, txt_processors, vis_proc_eval, txt_proc_eval, image_tensor
            gc.collect()
            torch.cuda.empty_cache()
            continue
        elif args.run_inference and (model_name, model_type) in retrieval_models:
            print(f"Run retrieval inference for {model_name} / {model_type}")

            classnames = ["kiwi", "apple", "dog", "cat", "nothing"]
            samples = {"image": image_tensor, "label": torch.zeros(1)}
            task_type = MultimodalClassificationTask
            dataset = argparse.Namespace(classnames=classnames, classtemplates="none")
            model = model.cuda()
            model.before_evaluation(dataset, task_type)
            out_dict = model.predict(samples)
            logits = out_dict["predictions"]
            targets = out_dict["targets"]  # this should just be the same as the input targets
            logits_cpu = logits.cpu()
            print(f"Logits: {logits_cpu}")
            print(f"Softmax (Temp.=1): {torch_stable_softmax(logits_cpu, temp=1.0)}")
            pred = logits_cpu.argmax(dim=-1).item()
            print(f"Prediction: {pred} {classnames[pred]}")

            del model, vis_processors, txt_processors, vis_proc_eval, txt_proc_eval, image_tensor
            gc.collect()
            torch.cuda.empty_cache()
            continue
        else:
            print(f"ERROR: No inference example found for {model_name} {model_type}")


if __name__ == "__main__":
    main()
