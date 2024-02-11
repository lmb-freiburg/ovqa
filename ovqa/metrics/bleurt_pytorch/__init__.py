"""
Note: Code was copied for simplicity, since bleurt_pytorch is not a pip package.

Code is licensed under Apache License Version 2.0.

Credits go to:

Luca Di Liello
luca.diliello@unitn.it
https://github.com/lucadiliello/bleurt-pytorch

Furthermore as mentioned in the README.md of the bleurt_pytorch git repository credits go to:

[Google original BLEURT](https://github.com/google-research/bleurt) implementation
[Transformers](https://huggingface.co/transformers) project
Users of this [issue](https://github.com/huggingface/datasets/issues/224)

"""

from ovqa.metrics.bleurt_pytorch.bleurt.configuration_bleurt import BleurtConfig  # noqa: F401
from ovqa.metrics.bleurt_pytorch.bleurt.modeling_bleurt import (
    BleurtForSequenceClassification,
)  # noqa: F401
from ovqa.metrics.bleurt_pytorch.bleurt.tokenization_bleurt import BleurtTokenizer  # noqa: F401
