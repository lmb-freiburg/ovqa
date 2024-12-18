from __future__ import annotations

import gc
import io
import lmdb
import numpy as np
import os
import torch
from loguru import logger
from math import ceil
from torch.cuda import OutOfMemoryError
from torch.utils.data import DataLoader, Dataset
from typing import Optional

from packg import format_exception
from ovqa.paths import get_cache_dir
from packg.tqdmext import tqdm_max_ncols
from packg.typext import PathType
from ovqa.metrics.llm.metric_prompts import MODES
from ovqa.metrics.preprocessing import get_preprocessing_fn, PrepC
from ovqa.metrics.torchmetrics_ext import MetricExt
from ovqa.models.llms.llm_loader import load_llm


class KVDataset(Dataset):
    def __init__(self, lmdb_file, n_cache):
        self.db: lmdb.Environment = None
        self.txn: lmdb.Transaction = None
        self.n_cache = n_cache
        self.lmdb_file = lmdb_file

    def __len__(self):
        return self.n_cache

    def ensure_db(self):
        # must create db link after workers is initialized, otherwise it will crash
        if self.db is None or self.txn is None:
            self.db = lmdb.open(self.lmdb_file.as_posix(), readonly=True)
            self.txn = self.db.begin(write=False)

    def __getitem__(self, idx):
        self.ensure_db()
        # with self.db.begin(write=False) as txn:
        txn = self.txn
        kv_data = txn.get(idx.to_bytes(4, "big"))
        kv_buffer = io.BytesIO(kv_data)
        try:
            past_key_values = torch.load(kv_buffer)
        except EOFError as e:
            raise EOFError(f"Corrupt index {idx} in file:\n{self.lmdb_file}\n") from e
        return past_key_values

    def close(self):
        self.txn.commit()
        self.db.close()


def collate_get0(batch):
    # run dataloader with batch_size 1 and use this collate to get the datapoint
    return batch[0]


def infinite_dataloader(dataloader):
    while True:
        for data in dataloader:
            yield data


def get_auto_batch_size(model_name: str):
    """
    Auto determine a good batch size for a model and GPU setup.

    Note: Currently only implemented for llama70b 4bit on either 2x24GB or 1x80GB.
    Assuming the metric produces ~400 tokens.

    Args:
        model_name:

    Returns:

    """
    current_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    num_gpus = torch.cuda.device_count()
    # assuming model is llama70b 4bit
    if current_mem_gb < 26:
        # running on 2x24GB
        batch_size = 32  # probably needs to be lowered alot for long outputs
    else:
        # running on 1x80GB
        batch_size = 32  # 64 broke for llama2 on long outputs (llava) at some point
    print(
        f"Auto-set batch size to {batch_size} given {num_gpus} GPU with {current_mem_gb:.1f}GB "
        f"and model {model_name}"
    )
    return batch_size


class LlamaMetricKV(MetricExt):
    """
    Notes for exllamav1 kernel
    - if max_tokens * batch_size is too high you may get "cublas not initialized" cuda error
    - if too low you may get "please use exllama_set_max_input_length to increase input length"

    """

    def __init__(
        self,
        n_incontext=5,
        preproc_cand=PrepC.NONE,
        preproc_ref=PrepC.NONE,
        preproc_question=PrepC.NONE,
        llama_name="llama2-70b-4bit",
        batch_size: Optional[int] = None,
        max_tokens: int = 2048,
        mode="vqa",
        verbose: bool = True,
        n_cache: int = 100,
        cache_dir: Optional[PathType] = None,
        map_size=100 * 1024**3,  # max size of KV database, increase if necessary
        seed=42,
    ) -> None:
        super().__init__()
        self.add_state("cands", default=[], dist_reduce_fx="cat")
        self.add_state("refs", default=[], dist_reduce_fx="cat")
        self.add_state("questions", default=[], dist_reduce_fx="cat")
        self.preproc_cand_fn = get_preprocessing_fn(preproc_ref)
        self.preproc_ref_fn = get_preprocessing_fn(preproc_cand)
        self.preproc_question_fn = get_preprocessing_fn(preproc_question)
        self.mode_info = MODES[mode]

        if batch_size is None:
            batch_size = get_auto_batch_size(llama_name)

        self.n_incontext = n_incontext
        self.llama_name = llama_name
        self.batch_size = batch_size
        self.mode = mode
        self.verbose = verbose
        self.n_cache = n_cache
        self.map_size = map_size
        self.seed = seed

        self._model = None
        self._tokenizer = None

        # for exllamav1 kernel we must estimate the total number of tokens in the input
        self.buffer_size = max_tokens * batch_size

        # preload kv cache
        if cache_dir is None:
            cache_dir = get_cache_dir()
        self.kv_cache_file = (
            cache_dir / "incontext_kvs" / f"{llama_name}~{mode}~{n_incontext}~{seed}.lmdb"
        )

    def actually_load_model(self):
        self._model, self._tokenizer = load_llm(self.llama_name, buffer_size=self.buffer_size)
        self.create_kv_cache()

    def get_model(self):
        if self._model is None:
            self.actually_load_model()
        return self._model

    def get_tokenizer(self):
        if self._tokenizer is None:
            self.actually_load_model()
        return self._tokenizer

    def close(self):
        self._model = None
        self._tokenizer = None
        torch.cuda.empty_cache()
        gc.collect()

    def update(
        self,
        cands: list[str],
        refs: list[str],
        *args: list[str],
    ) -> None:
        self.cands.extend(cands)
        self.refs.extend(refs)
        if len(args) == 0:
            pass
        elif len(args) == 1:
            self.questions.extend(args[0])
        else:
            raise ValueError(f"Too many arguments: {len(args)}")

    def parse_output(self, output_str: str, input_str="not_set") -> int:
        error_str = f"Model: {self.llama_name} input_str: {input_str} output_str: {output_str}"
        try:
            vote = int(output_str)
        except Exception as e:
            logger.error(f"{error_str}. No integer could be parsed:  {format_exception(e)}.")
            vote = 1
        if not 1 <= vote <= 5:
            logger.error(f"{error_str}. Vote {vote} not in 1-5.")
            vote = max(1, min(5, vote))
        return vote

    def compute(self) -> float:
        acc_list = self.compute_per_datapoint()
        return self.aggregate_per_datapoint_scores(acc_list)

    def create_kv_cache(self):
        # this is called from inside .get_model() so we shouldnt access that again (probably)
        model = self._model
        tokenizer = self._tokenizer
        print(f"Creating KV cache for model type {type(model)} in {self.kv_cache_file}")
        os.makedirs(self.kv_cache_file.parent, exist_ok=True)
        self.db: lmdb.Environment = lmdb.open(self.kv_cache_file.as_posix(), map_size=self.map_size)

        rng = np.random.default_rng(seed=self.seed)
        existing_ns = []
        with self.db.begin() as txn:
            txn: lmdb.Transaction
            cursor = txn.cursor()
            for n_bytes, _ in cursor:
                n = int.from_bytes(n_bytes, "big")
                existing_ns.append(n)

        if len(existing_ns) == 0:
            start_n = 0
        else:
            assert sorted(existing_ns) == list(range(len(existing_ns)))
            start_n = max(existing_ns) + 1
        print(f"Existing entries in KV cache: {start_n}")
        if start_n >= self.n_cache:
            print(f"Returning...")
            return

        print(f"Continuing...")
        incontext_data = self.mode_info["incontext_data"]
        with self.db.begin(write=True) as txn:
            for n in tqdm_max_ncols(list(range(start_n, self.n_cache)), desc="Create KV cache"):
                incontext_ids_sorted = np.arange(len(incontext_data))
                rng.shuffle(incontext_ids_sorted)
                incontext_ids = incontext_ids_sorted[: self.n_incontext]
                incontext_paragraph = []
                for e in range(self.n_incontext):
                    if e == 0 and "incontext_prompt_first" in self.mode_info:
                        incontext_prompt = self.mode_info["incontext_prompt_first"]
                    else:
                        incontext_prompt = self.mode_info["incontext_prompt"]
                    incontext_data_dict = incontext_data[incontext_ids[e]]
                    incontext_sentence = incontext_prompt.format(**incontext_data_dict)
                    incontext_paragraph.append(incontext_sentence)
                incontext_paragraph = "".join(incontext_paragraph)
                final_input = self.mode_info["intro"] + incontext_paragraph
                token_dict = tokenizer([final_input], return_tensors="pt", padding="longest").to(
                    model.device
                )
                with torch.inference_mode():
                    output_dict = model.forward(
                        input_ids=token_dict.input_ids,
                        attention_mask=token_dict.attention_mask,
                        return_dict=True,
                        use_cache=True,
                    )
                kvs = output_dict.past_key_values
                # tuple (length layers) x 2-tuple (keys, values) x
                # ([1=batch, 32=heads, 4=sequence, 128=head_dim])  # exact values depend on model

                n_bytes = n.to_bytes(4, "big")
                buffer = io.BytesIO()
                torch.save(kvs, buffer)
                txn.put(n_bytes, buffer.getvalue())

                if n < 2:
                    print(f"Prompt: '{final_input}'\nContext ids: {incontext_ids} ")
                    print(f"Input shape {token_dict.input_ids.shape}")
                    print(f"Record {n} / {n_bytes.hex()} cached: ")
                    print(f"{len(kvs)} x {len(kvs[0])} x {kvs[0][0].shape} ")
        print(f"Done creating KV cache")

    def compute_per_datapoint(self, return_dict=False) -> torch.Tensor:
        cands_raw = self.cands
        cands = [self.preproc_cand_fn(cand_raw) for cand_raw in cands_raw]
        refs_raw = self.refs
        refs = [self.preproc_ref_fn(ref_raw) for ref_raw in refs_raw]
        questions_raw = self.questions
        questions = [self.preproc_question_fn(question_raw) for question_raw in questions_raw]
        if len(cands) == 0:
            raise ValueError("No input candidates provided")
        use_questions = len(questions) > 0

        # access model to make it load
        model = self.get_model()
        tokenizer = self.get_tokenizer()

        # so now we want to process an entire batch with all the same KV cached values.
        # this means we must shuffle the input data instead of the KV cache
        shuffle_ids = np.arange(len(cands))
        rng = np.random.default_rng()
        rng.shuffle(shuffle_ids)
        unshuffle_ids = np.argsort(shuffle_ids)

        cands = [cands[i] for i in shuffle_ids]
        refs = [refs[i] for i in shuffle_ids]
        if use_questions:
            questions = [questions[i] for i in shuffle_ids]

        # currently workers > 0 is not possible since the KVs are directly loaded onto GPU
        dataset = KVDataset(self.kv_cache_file, self.n_cache)
        kv_loader = DataLoader(
            dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_get0
        )
        kv_loader_iter = iter(infinite_dataloader(kv_loader))
        all_votes = []
        for n_start in tqdm_max_ncols(
            range(0, len(cands), self.batch_size), total=ceil(len(cands) / self.batch_size)
        ):
            b_cands = cands[n_start : n_start + self.batch_size]
            b_refs = refs[n_start : n_start + self.batch_size]
            batch_size_now = len(b_cands)

            # kv_id = np.random.randint(0, self.n_cache)
            # kv_data = txn.get(kv_id.to_bytes(4, "big"))
            # kv_buffer = io.BytesIO(kv_data)
            # past_key_values = torch.load(kv_buffer)

            past_key_values = next(kv_loader_iter)

            # create input
            input_texts = []
            if use_questions:
                b_questions = questions[n_start : n_start + self.batch_size]
                for b in range(batch_size_now):
                    input_text = self.mode_info["output_prompt"].format(
                        question=b_questions[b],
                        candidate=b_cands[b],
                        reference=b_refs[b],
                    )
                    input_texts.append(input_text)
            else:
                for b in range(batch_size_now):
                    input_text = self.mode_info["output_prompt"].format(
                        candidate=b_cands[b],
                        reference=b_refs[b],
                    )
                    input_texts.append(input_text)

            # the <s> token already exists in the KV cache so skip it here
            token_dict = tokenizer(
                input_texts, return_tensors="pt", padding="longest", add_special_tokens=False
            ).to(model.device)

            # The attention_mask always needs to be: len(past_key_values) + len(input_ids)
            input_ids = token_dict["input_ids"]
            attention_mask = token_dict["attention_mask"]
            past_kv_len = past_key_values[0][0].shape[2]
            kv_mask = attention_mask.new_ones((batch_size_now, past_kv_len))
            new_attention_mask = torch.cat([kv_mask, attention_mask], dim=1)
            # print(
            #     f"Input ids: {input_ids.shape} Old atn: {attention_mask.shape} "
            #     f"New atn: {new_attention_mask.shape}"
            # )

            # kvs must be of len batch_size, so must be repeated (currently its len 1)
            # must be casted to list to be able to modify it
            past_key_values = list(past_key_values)
            for n_layer in range(len(past_key_values)):
                past_key_values[n_layer] = list(past_key_values[n_layer])
                for n_kv in range(2):
                    past_key_values[n_layer][n_kv] = past_key_values[n_layer][n_kv].expand(
                        batch_size_now, -1, -1, -1
                    )

            with torch.inference_mode():
                try:
                    logits = model.forward(
                        input_ids=input_ids,
                        attention_mask=new_attention_mask,
                        past_key_values=past_key_values,
                        use_cache=False,
                    ).logits
                except OutOfMemoryError as e:
                    print(f"Failed with input {input_ids.shape} {input_texts}")
                    os.system("nvidia-smi")
                    raise e
                pred_ids = logits[:, -1].argmax(-1)
                output_texts = tokenizer.batch_decode(pred_ids)

            output_votes = []
            for output_text in output_texts:
                output_vote = self.parse_output(output_text)
                output_votes.append(output_vote)
            all_votes.extend(output_votes)

            if self.verbose and n_start == 0:
                print(f"First datapoint, input: '{input_texts[0]}' output: '{output_texts[0]}'")

        # unshuffle
        all_votes = [all_votes[i] for i in unshuffle_ids]

        # now we have int list of 1-5 votes
        acc_tensor = (torch.tensor(all_votes, device=self.device, dtype=torch.float32) - 1) / 4
        if return_dict:
            return {"scores": acc_tensor}
        return acc_tensor


def main():
    metric = LlamaMetricKV(
        llama_name="llama2-7b-chat-4bit",
        n_cache=20,
    )

    questions = [
        "why is the sky blue",
        "how do plants make food",
        "what is the capital of France",
        "who wrote 'Romeo and Juliet'",
        "why do we sneeze",
        "what is photosynthesis",
    ]

    refs = [
        "light scattering",
        "photosynthesis",
        "Paris",
        "William Shakespeare",
        "to clear out irritants from the nose",
        "process by which plants convert light energy to chemical energy",
    ]

    cand = [
        "scattering of light",
        "conversion of sunlight into glucose",
        "Paris city",
        "Shakespeare",
        "we sneeze if our foot hurts",
        "conversion of light to energy in plants",
    ]
    metric.reset()
    for _ in range(20):  # 20*6 = 120 inputs to test some batch sizes
        metric.update(cand, refs, questions)
    scores = metric.compute_per_datapoint()
    print(scores)

    # test consistency
    scores = scores.reshape(20, -1)
    avg = torch.mean(scores, dim=0)
    deltas = ((scores - avg[None, :]).abs()).mean(dim=0)
    print("abs diff when running the same data with different incontext examples: ")
    print(deltas)

    # tensor([0.0000, 0.1687, 0.0000, 0.0000, 0.0000, 0.0000])  # seems stable enough


if __name__ == "__main__":
    main()
