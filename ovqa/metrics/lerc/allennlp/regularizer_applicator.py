import re
import torch
from typing import List, Tuple

from ovqa.metrics.lerc.allennlp.common.from_params import FromParams
from ovqa.metrics.lerc.allennlp.common.registrable import Registrable


class Regularizer(Registrable):
    """
    An abstract class representing a regularizer. It must implement
    call, returning a scalar tensor.
    """

    default_implementation = "l2"

    def __call__(self, parameter: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class RegularizerApplicator(FromParams):
    """
    Applies regularizers to the parameters of a Module based on regex matches.
    """

    def __init__(self, regexes: List[Tuple[str, Regularizer]] = None) -> None:
        """
        # Parameters

        regexes : `List[Tuple[str, Regularizer]]`, optional (default = `None`)
            A sequence of pairs (regex, Regularizer), where each Regularizer
            applies to the parameters its regex matches (and that haven't previously
            been matched).
        """
        self._regularizers = regexes or []

    def __call__(self, module: torch.nn.Module) -> torch.Tensor:
        """
        # Parameters

        module : `torch.nn.Module`, required
            The module to regularize.
        """
        accumulator = 0.0
        for name, parameter in module.named_parameters():
            # We first check if the parameter needs gradient updates or not
            if parameter.requires_grad:
                # For each parameter find the first matching regex.
                for regex, regularizer in self._regularizers:
                    if re.search(regex, name):
                        penalty = regularizer(parameter)
                        accumulator = accumulator + penalty
                        break
        return accumulator
