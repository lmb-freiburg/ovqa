"""
Note: removed loguru logger here since it kept deadlocking for some reason

sentence transformers https://huggingface.co/sentence-transformers
"""

import h5py
import math
import numpy as np
import os
import tempfile
import time
import torch
from PIL import Image
from attr import define
from datetime import datetime
from pathlib import Path

from packg import format_exception
from sentence_transformers.util import cos_sim
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer
from typing import Optional, Dict, Any, List, Union

from packg.constclass import Const
from ovqa.paths import get_data_dir
from packg.strings import quote_with_urlparse
from packg.typext import PathType
from ovqa.torchutils import count_params


class EmbeddingsPackageConst(Const):
    SENTENCE_TRANSFORMERS = "sentence-transformers"
    OPEN_CLIP = "open-clip"
    PROMCSE = "promcse"
    SIMCSE = "simcse"
    # PRECOMPUTED_NPZ = "precomputed-npz"


def get_sentence_embedder(
    model_name: str = "EVA01-g-14/laion400m_s11b_b41k",
    package_name: str = EmbeddingsPackageConst.OPEN_CLIP,
    # database specific settings
    use_db: bool = True,
    verbose: bool = True,
    compute_missing: bool = True,
    emb_dir: Optional[PathType] = None,
    save_to_db: bool = True,
    # embedder specific settings
    **kwargs,
) -> "SentenceEmbedderInterface":
    """
        Best model:
            get_sentence_embedder(
                model_name="EVA01-g-14/laion400m_s11b_b41k",
    #             package_name=EmbeddingsPackageConst.OPEN_CLIP,
            )

        Standard clip:
            get_sentence_embedder(model_name="clip-ViT-L-14",
            package_name=EmbeddingsPackageConst.SENTENCE_TRANSFORMERS,
            )

        OR:
            get_sentence_embedder(
                model_name="ViT-L-14/openai",
                package_name=EmbeddingsPackageConst.OPEN_CLIP,
            )


        Other good models:
            "gtr-t5-large" (313M) and slightly worse "all-mpnet-base-v2"(102M)
            (both 768-d)
    """
    if package_name == EmbeddingsPackageConst.SENTENCE_TRANSFORMERS:
        embedder = SentenceTransformersEmbedder.setup(package_name, model_name, **kwargs)
    elif package_name == EmbeddingsPackageConst.OPEN_CLIP:
        embedder = OpenClipEmbedder.setup(package_name, model_name, **kwargs)
    elif package_name == EmbeddingsPackageConst.PROMCSE:
        embedder = SentencePromCSEEmbedder.setup(package_name, model_name, **kwargs)
    elif package_name == EmbeddingsPackageConst.SIMCSE:
        embedder = SentenceSimCSEEmbedder.setup(package_name, model_name, **kwargs)
    else:
        raise ValueError(f"Unknown package name {package_name}")

    if not use_db:
        return embedder
    db_embedder = SentenceEmbedderWithDb.wrap_embedder(
        embedder,
        verbose=verbose,
        compute_missing=compute_missing,
        emb_dir=emb_dir,
        save_to_db=save_to_db,
    )
    return db_embedder


def normalize_embeddings(embedding_arr, tensor_type: str = "numpy"):
    """
    normalize embeddings to unit length
    """
    eps = 1e-8
    err_msg = "Some embeddings have zero length! Run tools/fix_embeddings.py"
    if tensor_type == "numpy":
        denom = np.linalg.norm(embedding_arr, axis=-1, keepdims=True)
        assert np.all(denom > eps), err_msg
        return embedding_arr / denom
    if tensor_type == "torch":
        denom = torch.norm(embedding_arr, dim=-1, keepdim=True)
        assert torch.all(denom > eps), err_msg
        return embedding_arr / denom
    raise ValueError(f"Unknown tensor type {tensor_type}")


@define(slots=False)
class SentenceEmbedderInterface:
    package_name: str
    model_name: str

    @classmethod
    def setup(cls, **kwargs):
        raise NotImplementedError

    @property
    def model(self):
        # some methods may not need a model
        raise NotImplementedError

    def encode(
        self, sentences: List[str], normalize: bool = False, return_type: str = "numpy"
    ) -> Union[np.ndarray, torch.Tensor]:
        """

        Args:
            sentences: List of sentences
            normalize: If True, L2-normalize the embeddings to unit length
            return_type: "numpy" or "torch"

        Returns:
            numpy array shape (N_sentences, D_embedding_dimensions)
        """
        raise NotImplementedError

    def encode_visual(self, image):
        raise NotImplementedError

    def encode_visuals_from_files(
        self,
        imagefiles: PathType,
        base_dir=None,
        normalize: bool = False,
        return_type: str = "numpy",
    ):
        raise NotImplementedError

    def close(self):
        """Unload models from GPU to free up the space"""
        pass


@define(slots=False)
class SentenceTransformersEmbedder(SentenceEmbedderInterface):
    device: str = "cuda"
    batch_size: int = 32
    _model: Any = None

    @classmethod
    def setup(
        cls,
        package_name: str,
        model_name: str,
        device: str = "cuda",
        batch_size: int = 32,
    ):
        return cls(
            package_name=package_name,
            model_name=model_name,
            device=device,
            batch_size=batch_size,
        )

    @property
    def model(self):
        # only load model if needed
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode(
        self, sentences: List[str], normalize: bool = False, return_type: str = "numpy"
    ) -> Union[np.ndarray, torch.Tensor]:
        try:
            outputs = self.model.encode(
                sentences,
                batch_size=self.batch_size,
                device=self.device,
                convert_to_tensor=return_type == "torch",
            )
        except RuntimeError as e:
            if not str(e).startswith("The size of tensor a"):
                # another error
                raise e
            # sentence transformers clip cannot handle too long sentences
            # get the config of the clip model and find the maximum length
            config = self.model._modules["0"].model.config  # noqa
            max_len = config.text_config.max_position_embeddings - 1

            # tokenize the sentences, cut them
            lens = self.model.tokenize(sentences)["attention_mask"].sum(-1).tolist()
            new_sentences = []
            for old_sentence, old_len in zip(sentences, lens):
                cut_sentence = _cut_sentence(old_sentence, self.model.tokenize, max_len, old_len)
                new_sentences.append(cut_sentence)
                assert (
                    cut_sentence == old_sentence[: len(cut_sentence)]
                ), f"Mismatch after tokenizer roundtrip: {old_sentence} vs {cut_sentence}"
            outputs = self.model.encode(
                new_sentences,
                batch_size=self.batch_size,
                device=self.device,
                convert_to_tensor=return_type == "torch",
            )

        if normalize:
            outputs = normalize_embeddings(outputs, tensor_type=return_type)
        return outputs

    def close(self):
        self._model = None


def _cut_sentence(old_sentence, tokenize_fn, max_len, old_len):
    if old_len < max_len:
        return old_sentence

    # dont want to change the sentence and the tokenizer is not roundtrip-safe
    # simply reduce the string length until it fits
    for cut_i in range(1, len(old_sentence)):
        cut_sentence = old_sentence[:-cut_i]
        new_len = tokenize_fn([cut_sentence])["attention_mask"].shape[-1]
        if new_len < max_len:
            print(
                f"ERROR: Cutting too long sentence {old_sentence} with {len(old_sentence)} chars, "
                f"{old_len} tokens to {len(cut_sentence)} chars, {new_len} tokens."
            )
            break
    else:
        raise RuntimeError(f"Could not cut sentence {old_sentence} " f"to fit into {max_len}")
    return cut_sentence


@define(slots=False)
class OpenClipEmbedder(SentenceEmbedderInterface):
    """
    modelname should be $model/$pretrained_dataset e.g.
        ViT-L-14/openai

    """

    device: str = "cuda"
    batch_size: int = 32
    enable_vision: bool = False
    image_size: int = 224
    _model: Any = None
    _tokenizer: Any = None
    _pad_token_id: int = None

    @classmethod
    def setup(
        cls,
        package_name: str,
        model_name: str,
        device: str = "cuda",
        batch_size: int = 32,
        enable_vision: bool = False,
        image_size: int = 224,
    ):
        return cls(
            package_name=package_name,
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            enable_vision=enable_vision,
            image_size=image_size,
        )

    @property
    def model(self):
        if self._model is None:
            import open_clip

            _ = open_clip.model.CLIP  # reference
            model_type, model_pretrained = self.model_name.split("/")
            # load only on CPU, then delete the non-text part, then move.
            model = open_clip.create_model(model_type, model_pretrained, device="cpu").eval()
            total_params_full = count_params(model) / 1e9
            vis_params = count_params(model.visual) / 1e9
            text_params = total_params_full - vis_params
            if self.enable_vision:
                from ovqa.processors.openclip_processors import OpenClipImageEvalProcessor

                self.visual_pre = OpenClipImageEvalProcessor(
                    dict(
                        model_name=model_type,
                        pretrained_name=model_pretrained,
                        image_size=self.image_size,
                    )
                )
                print(f"P{self.visual_pre}")
            else:
                del model.visual
            total_params = count_params(model) / 1e9
            print(
                f"Setup embedding model {model_type}/{model_pretrained} with params: "
                f"{total_params:.2f}G total / {vis_params:.2f}G vision / "
                f"{text_params:.2f}G text."
                f"Final model with {total_params:.2f}G params, {self.enable_vision=}"
            )
            model = model.to(self.device)
            # count params for vis and text and print them first
            # self.device
            self._model = model
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            import open_clip

            model_type, model_pretrained = self.model_name.split("/")
            tokenizer = open_clip.get_tokenizer(model_type)
            # hack to get the token ids. "" -> [[sot, eot, pad, pad, ...]]
            empty_seq = tokenizer("")
            sot_token_id, eot_token_id, pad_token_id = (int(empty_seq[0, i]) for i in [0, 1, -1])
            # print(
            #     f"Created open clip tokenizer with sot={sot_token_id}, "
            #     f"eot={eot_token_id}, pad={pad_token_id}"
            # )
            self._pad_token_id = pad_token_id
            self._tokenizer = tokenizer
        return self._tokenizer

    def encode(
        self, sentences: List[str], normalize: bool = False, return_type: str = "numpy"
    ) -> Union[np.ndarray, torch.Tensor]:
        device = self.device
        model = self.model
        tokenizer = self.tokenizer

        # manually batchify
        n_batches = math.ceil(len(sentences) / self.batch_size)

        feature_collector = []
        for n_batch in range(n_batches):
            start_pos, end_pos = (
                n_batch * self.batch_size,
                (n_batch + 1) * self.batch_size,
            )
            sentence_batch = sentences[start_pos:end_pos]
            tokens = tokenizer(sentence_batch).to(device)
            with torch.no_grad():
                text_features = model.encode_text(tokens)
                if normalize:
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                feature_collector.append(text_features.cpu())
        text_features = torch.cat(feature_collector, dim=0)

        if return_type == "numpy":
            text_features = text_features.numpy()
        return text_features

    @torch.no_grad()
    def encode_visual(self, image: Image.Image):
        """Not batched for now"""
        assert self.enable_vision
        _ = self.model  # must make sure the model is lazy loaded
        image_torch = self.visual_pre(image)
        return self.model.encode_image(image_torch.to(self.device)[None], normalize=True).cpu()[0]

    def close(self):
        self._model = None
        self._tokenizer = None


class Namespace(object):
    def __init__(self, model_name_or_path: str, pooler_type: str, pre_seq_len: int):
        self.model_name_or_path = model_name_or_path
        self.pooler_type = pooler_type
        self.pre_seq_len = pre_seq_len

        self.temp = None
        self.hard_negative_weight = None
        self.do_mlm = None
        self.mlm_weight = None
        self.mlp_only_train = None
        self.prefix_projection = None
        self.prefix_hidden_size = None
        self.do_eh_loss = None
        self.eh_loss_margin = None
        self.eh_loss_weight = None
        self.cache_dir = None
        self.use_auth_token = None
        pass


@define(slots=False)
class SentencePromCSEEmbedder(SentenceEmbedderInterface):
    #  pip install promcse --no-deps
    device: str = "cuda"
    batch_size: int = 32
    _model: Any = None
    model_args: dict = {}
    tokenizer: Any = None

    @classmethod
    def setup(
        cls,
        package_name: str,
        model_name: str,
        device: str = "cuda",
        batch_size: int = 32,
        pooler_type: str = "cls",
        pre_seq_len: int = 40,
    ):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_args = Namespace(model_name, pooler_type, pre_seq_len)

        return cls(
            package_name=package_name,
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            model_args=model_args,
            tokenizer=tokenizer,
        )

    @property
    def model(self):
        # only load model if needed
        if self._model is None:
            from promcse.models import BertForCL, RobertaForCL

            if "roberta" in self.model_name:
                self._model = RobertaForCL.from_pretrained(
                    self.model_name,
                    from_tf=bool(".ckpt" in self.model_name),
                    config=AutoConfig.from_pretrained(self.model_name),
                    cache_dir=self.model_args.cache_dir,
                    revision="main",
                    use_auth_token=True if self.model_args.use_auth_token else None,
                    model_args=self.model_args,
                ).to(self.device)
            elif "bert" in self.model_name:
                self._model = BertForCL.from_pretrained(
                    self.model_name,
                    from_tf=bool(".ckpt" in self.model_name),
                    config=AutoConfig.from_pretrained(self.model_name),
                    cache_dir=self.model_args.cache_dir,
                    revision="main",
                    use_auth_token=True if self.model_args.use_auth_token else None,
                    model_args=self.model_args,
                ).to(self.device)
            else:
                assert False, f"Model for evaluation metric not defined {self.model_name}"
        return self._model

    def encode(
        self, sentences: List[str], normalize: bool = False, return_type: str = "numpy"
    ) -> Union[np.ndarray, torch.Tensor]:
        """

        Args:
            sentences: List of sentences
            normalize: If True, L2-normalize the embeddings to unit length
            return_type: "numpy" or "torch"

        Returns:
            numpy array shape (N_sentences, D_embedding_dimensions)
        """

        embedding_list = []
        with torch.no_grad():
            total_batch = len(sentences) // self.batch_size + (
                1 if len(sentences) % self.batch_size > 0 else 0
            )
            for batch_id in range(total_batch):
                inputs = self.tokenizer(
                    sentences[batch_id * self.batch_size : (batch_id + 1) * self.batch_size],
                    padding=True,
                    truncation=True,
                    max_length=self.model_args.pre_seq_len,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                embeddings = self.model(
                    **inputs, output_hidden_states=True, return_dict=True, sent_emb=True
                ).pooler_output
                if normalize:
                    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
                embedding_list.append(embeddings.cpu())
        embeddings = torch.cat(embedding_list, 0)

        if return_type == "numpy":
            return embeddings.numpy()

        return embeddings

    def close(self):
        self._model = None


@define(slots=False)
class SentenceSimCSEEmbedder(SentenceEmbedderInterface):
    """
    Models Available
    diffcse
    voidism/diffcse-bert-base-uncased-sts
    voidism/diffcse-bert-base-uncased-trans
    voidism/diffcse-roberta-base-sts
    voidism/diffcse-roberta-base-trans
    simcse
    princeton-nlp/unsup-simcse-bert-base-uncased
    princeton-nlp/unsup-simcse-bert-large-uncased
    princeton-nlp/unsup-simcse-roberta-base
    princeton-nlp/unsup-simcse-roberta-large
    princeton-nlp/sup-simcse-bert-base-uncased
    princeton-nlp/sup-simcse-bert-large-uncased
    princeton-nlp/sup-simcse-roberta-base
    princeton-nlp/sup-simcse-roberta-large
    """

    #  pip install simcse --no-deps
    device: str = "cuda"
    batch_size: int = 32
    _model: Any = None
    model_args: dict = {}
    tokenizer: Any = None

    @classmethod
    def setup(
        cls,
        package_name: str,
        model_name: str,
        device: str = "cuda",
        batch_size: int = 32,
        num_cells: int = 100,
        num_cells_in_search: int = 10,
        pooler=None,
        max_length: int = 128,
    ):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_args = {
            "model_name": model_name,
            "num_cells": num_cells,
            "num_cells_in_search": num_cells_in_search,
            "pooler": pooler,
            "max_length": max_length,
        }

        return cls(
            package_name=package_name,
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            model_args=model_args,
            tokenizer=tokenizer,
        )

    @property
    def model(self):
        # only load model if needed
        if self._model is None:
            from transformers import AutoModel

            if "bert" in self.model_name:
                self._model = AutoModel.from_pretrained(self.model_name).to(self.device)
            else:
                assert False, f"Model for evaluation metric not defined {self.model_name}"

            self.index = None
            self.is_faiss_index = False
            self.num_cells = self.model_args["num_cells"]
            self.num_cells_in_search = self.model_args["num_cells_in_search"]

            if self.model_args["pooler"] is not None:
                self.pooler = self.model_args["pooler"]
            elif "unsup" in self.model_name:
                print(
                    "Use `cls_before_pooler` for unsupervised models. If you want to use other pooling policy, specify `pooler` argument."
                )
                self.pooler = "cls_before_pooler"
            else:
                self.pooler = "cls"
        return self._model

    def encode(
        self, sentences: List[str], normalize: bool = False, return_type: str = "numpy"
    ) -> Union[np.ndarray, torch.Tensor]:
        """

        Args:
            sentences: List of sentences
            normalize: If True, L2-normalize the embeddings to unit length
            return_type: "numpy" or "torch"

        Returns:
            numpy array shape (N_sentences, D_embedding_dimensions)
        """

        embedding_list = []
        with torch.no_grad():
            total_batch = len(sentences) // self.batch_size + (
                1 if len(sentences) % self.batch_size > 0 else 0
            )
            for batch_id in range(total_batch):
                inputs = self.tokenizer(
                    sentences[batch_id * self.batch_size : (batch_id + 1) * self.batch_size],
                    padding=True,
                    truncation=True,
                    max_length=self.model_args["max_length"],
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs, return_dict=True)
                if self.pooler == "cls":
                    embeddings = outputs.pooler_output
                elif self.pooler == "cls_before_pooler":
                    embeddings = outputs.last_hidden_state[:, 0]
                else:
                    raise NotImplementedError
                if normalize:
                    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
                embedding_list.append(embeddings.cpu())
        embeddings = torch.cat(embedding_list, 0)

        if return_type == "numpy":
            return embeddings.numpy()

        return embeddings

    def close(self):
        self._model = None


@define(slots=False)
class SentenceEmbedderWithDb(SentenceEmbedderInterface):
    """
    H5 wrapper for text embeddings.

    See function get_sentence_embedder() to create this class.
    """

    embedder: SentenceEmbedderInterface
    verbose: bool
    compute_missing: bool
    save_to_db: bool
    h5_file: Path = None
    h5_file_visual: Path = None

    @property
    def model(self):
        return self.embedder.model

    @classmethod
    def wrap_embedder(
        cls,
        embedder: SentenceEmbedderInterface,
        verbose: bool = True,
        compute_missing: bool = True,
        emb_dir: Optional[PathType] = None,
        save_to_db: bool = True,
    ):
        emb_dir = get_data_dir() / "text_embeddings/sentences" if emb_dir is None else Path(emb_dir)
        model_name = embedder.model_name
        model_name_safe = model_name.replace("/", "~")
        package_name = embedder.package_name
        h5_file = emb_dir / f"{package_name}~{model_name_safe}.h5"
        if verbose:
            print(f"Embedder {model_name} with cache file {h5_file}")
        h5_file_visual = emb_dir / f"{package_name}~{model_name_safe}~vis.h5"
        return cls(
            package_name=package_name,
            model_name=model_name,
            embedder=embedder,
            verbose=verbose,
            compute_missing=compute_missing,
            h5_file=h5_file,
            h5_file_visual=h5_file_visual,
            save_to_db=save_to_db,
        )

    def encode(
        self, sentences: List[str], normalize: bool = False, return_type: str = "numpy"
    ) -> Union[np.ndarray, torch.Tensor]:
        missing_sentences_set = set(sentences)

        feature_dict = {}
        if self.h5_file.is_file():
            # read existing embeddings from h5 file as numpy arrays
            with h5py.File(self.h5_file, "r", libver="latest", swmr=True) as f:
                #  NEVER do set(f.keys()) its too slow
                for sentence in list(missing_sentences_set):
                    quoted_sentence = quote_with_urlparse(sentence, prefix="q")
                    if quoted_sentence in f:
                        # f[quoted_sentence].refresh()  # "datasets" (features) never change
                        feature_dict[sentence] = np.array(f[quoted_sentence])
                        missing_sentences_set.remove(sentence)

        if len(missing_sentences_set) > 0:
            if not self.compute_missing:
                raise ValueError(
                    f"missing {len(missing_sentences_set)} embeddings in {self.h5_file} "
                )

            # compute missing embeddings
            sentences_to_compute = sorted(missing_sentences_set)
            outputs: np.ndarray = self.embedder.encode(
                sentences_to_compute, normalize=False, return_type="numpy"
            )
            feature_dict_u = {}
            for i, sentence in enumerate(sentences_to_compute):
                feat = outputs[i]
                feature_dict_u[sentence] = feat

            if self.save_to_db:
                # save embeddings to db
                # single write multi read - use lockfile to make sure only one process writes
                os.makedirs(self.h5_file.parent, exist_ok=True)
                lockfile = self.h5_file.parent / f"{self.h5_file.name}.lock"
                while lockfile.is_file():
                    print(
                        f"Waiting for lockfile to be deleted at {lockfile} with content "
                        f"{lockfile.read_text(encoding='utf-8')}. If this lockfile is leftover "
                        f"from a crash, delete it. If another process is writing to this h5 file "
                        f"at the moment, wait for it to finish."
                    )
                    time.sleep(10)

                while True:
                    try:
                        with h5py.File(self.h5_file, "a", libver="latest") as f:
                            lockfile.write_text(f"locked at {datetime.now()}", encoding="utf-8")
                            f.swmr_mode = True
                            for sentence, feat in feature_dict_u.items():
                                quoted_sentence = quote_with_urlparse(sentence, prefix="q")
                                f.create_dataset(quoted_sentence, data=feat)
                            f.flush()
                            break
                    except BlockingIOError as e:
                        print(f"Waiting for file unlock of {self.h5_file}... {format_exception(e)}")
                        time.sleep(10)
                lockfile.unlink()

            # add newly computed embeddings to the output dict
            feature_dict.update(feature_dict_u)

        # convert output dict to tensor
        outputs = prepare_output(sentences, feature_dict, return_type=return_type)
        if normalize:
            outputs = normalize_embeddings(outputs, tensor_type=return_type)
        return outputs

    def encode_visual(self, image):
        return self.embedder.encode_visual(image)

    def encode_visuals_from_files(
        self,
        imagefiles: List[PathType],
        base_dir=None,
        normalize: bool = False,
        return_type: str = "numpy",
    ):
        """If given a list of filenames we can actually cache the results
        todo merge this and the encode function above, batchify / add dataloader
        """
        imagefiles = [Path(imagefile).as_posix() for imagefile in imagefiles]
        missing_imagefiles_set = set(imagefiles)
        feature_dict = {}
        if self.h5_file_visual.is_file():
            # read existing embeddings from h5 file as numpy arrays
            with h5py.File(self.h5_file_visual, "r", libver="latest", swmr=True) as f:
                for imagefile in list(missing_imagefiles_set):
                    quoted_imagefile = quote_with_urlparse(imagefile, prefix="f")
                    if quoted_imagefile in f:
                        # f[quoted_sentence].refresh()  # "datasets" (features) never change
                        feature_dict[imagefile] = np.array(f[quoted_imagefile])
                        missing_imagefiles_set.remove(imagefile)

        if len(missing_imagefiles_set) > 0:
            if not self.compute_missing:
                raise ValueError(
                    f"missing {len(missing_imagefiles_set)} embeddings in {self.h5_file_visual} "
                )

            # compute missing embeddings
            imagefiles_to_compute = sorted(missing_imagefiles_set)
            embs = []
            for imagefile in tqdm(
                imagefiles_to_compute, desc="computing visual embeddings", disable=not self.verbose
            ):
                full_imagefile = Path(base_dir) / imagefile if base_dir is not None else imagefile
                image = Image.open(full_imagefile).convert("RGB")
                emb = self.encode_visual(image).cpu().numpy()
                embs.append(emb)
            outputs = np.stack(embs, axis=0)

            feature_dict_u = {}
            for i, imagefile in enumerate(imagefiles_to_compute):
                feat = outputs[i]
                feature_dict_u[imagefile] = feat

            if self.save_to_db:
                # save embeddings to db
                # single write multi read - use lockfile to make sure only one process writes
                os.makedirs(self.h5_file_visual.parent, exist_ok=True)
                lockfile = self.h5_file_visual.parent / f"{self.h5_file_visual.name}.lock"
                assert (
                    not lockfile.is_file()
                ), f"lockfile {lockfile} exists with content {lockfile.read_text(encoding='utf-8')}"
                lockfile.write_text(f"locked at {datetime.now()}", encoding="utf-8")

                with h5py.File(self.h5_file_visual, "a", libver="latest") as f:
                    f.swmr_mode = True
                    for imagefile, feat in feature_dict_u.items():
                        quoted_imagefile = quote_with_urlparse(imagefile, prefix="f")
                        f.create_dataset(quoted_imagefile, data=feat)
                    f.flush()
                lockfile.unlink()

            # add newly computed embeddings to the output dict
            feature_dict.update(feature_dict_u)

        # convert output dict to tensor
        outputs = prepare_output(imagefiles, feature_dict, return_type=return_type)
        if normalize:
            outputs = normalize_embeddings(outputs, tensor_type=return_type)
        return outputs


def prepare_output(
    sentences: List[str],
    feature_dict: Dict[str, np.ndarray],
    return_type: str = "numpy",
) -> Union[np.ndarray, torch.Tensor]:
    feature_list = [feature_dict[s] for s in sentences]
    feature_stack = np.stack(feature_list, axis=0)

    lengths = np.linalg.norm(feature_stack, axis=-1, keepdims=False)
    eps = 1e-8
    if np.any(lengths < eps):
        wrong_ids = np.where(lengths < eps)[0].tolist()
        for wrong_id in wrong_ids:
            print(
                f"ERROR: Length {lengths[wrong_id]} too short, sentence '{sentences[wrong_id]}', id {wrong_id}"
            )
        raise RuntimeError(f"See log")

    if return_type == "numpy":
        return feature_stack
    if return_type == "torch":
        return torch.from_numpy(feature_stack)
    raise ValueError(f"Unknown return_type {return_type}")


def main():
    with tempfile.TemporaryDirectory() as tmpdirname:
        embedder = get_sentence_embedder(
            emb_dir=tmpdirname,
            package_name=EmbeddingsPackageConst.OPEN_CLIP,
            model_name="ViT-L-14/openai",
            batch_size=2,
        )
        outputs = embedder.encode(["hello world", "goodbye my friends", "ice cream"])
        print(type(outputs), outputs.shape, outputs.dtype)
        print(cos_sim(outputs, outputs))

    with tempfile.TemporaryDirectory() as tmpdirname:
        embedder = get_sentence_embedder(emb_dir=tmpdirname)
        outputs = embedder.encode(["hello world", "goodbye my friends", "ice cream"])
        print(type(outputs), outputs.shape, outputs.dtype)
        print(cos_sim(outputs, outputs))

    with tempfile.TemporaryDirectory() as tmpdirname:
        embedder = get_sentence_embedder(emb_dir=tmpdirname)
        outputs = embedder.encode(
            ["hello world", "goodbye my friends", "ice cream"], return_type="torch"
        )
        print(type(outputs), outputs.shape, outputs.dtype)
        print(cos_sim(outputs, outputs))

        embedder.encode(["a way too long sentence " * 100])


if __name__ == "__main__":
    main()
