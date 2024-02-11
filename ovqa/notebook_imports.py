from collections import defaultdict, Counter

import base64
import hashlib
import io
import json
import numpy as np
import os
import random
import re
import shutil
import sys
import time
import torch
from IPython.display import Image, display, HTML
from PIL import Image
from copy import deepcopy
from enum import Enum
from loguru import logger
from matplotlib import pyplot as plt
from pathlib import Path
from pprint import pprint, pformat
from sentence_transformers.util import cos_sim
from timeit import default_timer as timer
from torch import nn
from tqdm import tqdm
from typing import (
    Dict,
    List,
    Optional,
    Any,
    Iterable,
    Mapping,
    Tuple,
    Union,
    Callable,
    BinaryIO,
    Sequence,
    Collection,
)

from ovqa.common.lavis.config import Config
from ovqa.datasets.interface_metadata import ClsMetadataInterface
from ovqa.datasets.meta_loading import meta_loader
from packg import format_exception
from packg.iotools.jsonext import load_json, dump_json, loads_json, dumps_json
from packg.log import configure_logger
from packg.magic import reload_recursive
from packg.paths import (
    get_cache_dir,
    get_data_dir,
)
from packg.strings import b64_encode_from_bytes
from visiontext.htmltools import NotebookHTMLPrinter, display_html_table
from visiontext.images import PILImageScaler

# # the __all__ list below is used to stop pycharm or other tools from removing unused imports
# # to update it after changing the imports above, uncomment the code below and copypaste the output
# imported_modules = [m for m in globals().keys() if not m.startswith("_")]
# print(f"__all__ = {repr(imported_modules)}")

__all__ = [
    "base64",
    "hashlib",
    "io",
    "json",
    "os",
    "random",
    "shutil",
    "sys",
    "time",
    "deepcopy",
    "Enum",
    "Path",
    "pprint",
    "timer",
    "Dict",
    "List",
    "Optional",
    "Any",
    "Iterable",
    "Mapping",
    "Tuple",
    "Union",
    "Callable",
    "BinaryIO",
    "Sequence",
    "Collection",
    "plt",
    "np",
    "torch",
    "Image",
    "display",
    "HTML",
    "logger",
    "defaultdict",
    "Counter",
    "load_json",
    "dump_json",
    "loads_json",
    "dumps_json",
    "get_cache_dir",
    "get_data_dir",
    "b64_encode_from_bytes",
    "reload_recursive",
    "Config",
    "nn",
    "tqdm",
    "NotebookHTMLPrinter",
    "display_html_table",
    "ClsMetadataInterface",
    "meta_loader",
    "PILImageScaler",
    "cos_sim",
    "format_exception",
    "re",
    "pformat",
    "configure_logger",
]
