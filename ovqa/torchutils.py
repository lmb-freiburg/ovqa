from typing import Dict


def _group_by_requires_grad(param):
    if param.requires_grad:
        return "grad"
    return "no_grad"


def _get_params(parameters):
    # support inputs: model, model.parameters(), model.named_parameters()
    if hasattr(parameters, "values"):
        parameters = parameters.values()
    if hasattr(parameters, "parameters"):
        parameters = parameters.parameters()
    return parameters


def count_params_by_requires_grad(parameters) -> Dict[str, int]:
    parameters = _get_params(parameters)
    groups = {"grad": 0, "no_grad": 0}
    for v in parameters:
        if v.requires_grad:
            groups["grad"] += v.numel()
        else:
            groups["no_grad"] += v.numel()
    return groups


def count_params(parameters) -> int:
    parameters = _get_params(parameters)
    total = 0
    for v in parameters:
        total += v.numel()
    return total
