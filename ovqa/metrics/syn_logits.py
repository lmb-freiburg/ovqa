import torch


def get_name2syn_from_classes(
    class_names: list[str],
    synonym_dict: dict[str, int] = None,
) -> dict[str, list[str]]:
    """

    Args:
        class_names:  list of class names
        synonym_dict: dictionary of synonyms to class indices

    Returns:
        dict of class name to list of synonyms
    """
    if synonym_dict is None:
        synonym_dict = {}
    cls_name2syn = {name: [name] for name in class_names}
    for syn, clsidx in synonym_dict.items():
        if isinstance(clsidx, str) or isinstance(clsidx, int):
            cls_name2syn[class_names[clsidx]].append(syn)
        elif isinstance(clsidx, list):
            for kclsidx in clsidx:
                cls_name2syn[class_names[kclsidx]].append(syn)
    return cls_name2syn


def reduce_synonym_logits_over_classes(
    logits: torch.Tensor, classids: list[int], arg_max_or_average_syn: str = "arg_max_syn"
) -> torch.Tensor:
    """
    map synonym logits to class logits

    Args:
        logits: shape(n_datapoints, n_synonyms)
        classids: shape(n_synonyms) in [0, n_classes) e.g. [0, 0, 1, 2, 2, 2, ...]
        arg_max_or_average_syn: "arg_max_syn" or "average_syn"

    Returns:
        logits: shape(n_datapoints, n_classes)

    """
    # make sure the classids are a proper map from synonyms to classes
    assert classids == sorted(classids)
    set_classids = set(classids)
    num_classes = len(set_classids)
    assert sorted(set_classids) == list(range(num_classes))

    # invert the classids to get the end of the range of synonyms for each class
    # e.g. get [2, 3, 6, ...] from [0, 0, 1, 2, 2, 2, ...]
    classids = torch.tensor(classids)
    end_bounds = torch.empty(num_classes, dtype=torch.int64)
    end_bounds[:-1] = torch.where(classids[1:] - classids[:-1])[0] + 1
    end_bounds[-1] = len(classids)
    end_bounds = end_bounds.tolist()

    # map predictions over the set of synonyms back to the smaller set of classes
    # by taking the max or average syn
    logits_new = logits.new_zeros((logits.shape[0], num_classes))
    start_bound = 0
    for i, end_bound in enumerate(end_bounds):
        # logits for class i is the maximum logit of synonym logits
        if arg_max_or_average_syn == "average_syn":
            logits_new[:, i] = torch.mean(logits[:, start_bound:end_bound], dim=1)
        elif arg_max_or_average_syn == "arg_max_syn":
            logits_new[:, i] = torch.max(logits[:, start_bound:end_bound], dim=1).values
        else:
            raise ValueError(
                f"arg_max_or_average_syn must be 'average_syn' or 'arg_max_syn', "
                f"got {arg_max_or_average_syn}"
            )
        start_bound = end_bound
    logits = logits_new
    return logits


def reduce_synonym_logits_over_predictions(
    logits: torch.Tensor, pred_syn_ids: list[int], arg_max_or_average_syn: str = "average_syn"
) -> torch.Tensor:
    """
    Same as above but reducing synonyms on the datapoint axis (0) instead of the class axis (1)
    simply by transposing the logits twice
    """
    return reduce_synonym_logits_over_classes(
        logits=logits.T, classids=pred_syn_ids, arg_max_or_average_syn=arg_max_or_average_syn
    ).T
