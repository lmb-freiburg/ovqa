from collections import defaultdict

import logging
import re
from pprint import pformat


def apply_layer_decay(
    model,
    lr_layer_decay: float,
    lr_layer_decay_regex_base: list[str],
    lr_layer_decay_regex_layers: list[str],
):
    """
    Calculate learning rates for lr decay

    Args:
        model:
        lr_layer_decay:
        lr_layer_decay_regex_base: regex to find the base (input) of the tower to apply lr decay
            e.g. ["visual_encoder[.]cls_token", "visual_encoder[.]pos_embed", "visual_encoder[.]patch_embed[.].*"]
        lr_layer_decay_regex_layers:  regex to find the layer weights and layer number
            notice the capturing group ([0-9]+) in the regex to find the layer number
            e.g. ["visual_encoder[.]blocks[.]([0-9]+)[.].*"]
    Returns:

    """
    logging.info(f"Using layer-wise learning rate decay {lr_layer_decay}.")
    res_base = [re.compile(rx) for rx in lr_layer_decay_regex_base]
    res_layers = [re.compile(rx) for rx in lr_layer_decay_regex_layers]
    groups_layers = defaultdict(list)
    group_base, group_other = [], []
    named_params_dict = {}
    num_parameters, num_untrained = 0, 0
    for n, p in model.named_parameters():
        if not p.requires_grad:
            num_untrained += p.data.nelement()
            continue  # frozen weights
        num_parameters += p.data.nelement()
        named_params_dict[n] = p
        for rx in res_base:
            if rx.fullmatch(n):
                group_base.append(n)
                break
        else:
            for rx in res_layers:
                match = rx.fullmatch(n)
                if match is None:
                    continue
                layer_num = int(match.group(1))
                groups_layers[layer_num].append(n)
                break
            else:
                group_other.append(n)

    num_layers = len(groups_layers)
    assert sorted(groups_layers.keys()) == list(
        range(num_layers)
    ), f"Layers must be numbered 0, 1, 2, ... but are {groups_layers.keys()}"
    # sorted into base of the tower, layers, and top of the tower (output)
    # we can ignore the top, this will have full LR
    # now for each layer to go backward from the top, multiply the LR by the decay
    param_name_groups = {}
    for n_layer_from_top in range(num_layers):
        n_layer = num_layers - 1 - n_layer_from_top
        lr_mult = lr_layer_decay ** (n_layer_from_top + 1)
        logging.info(f"Layer {n_layer} LR mult {lr_mult}")
        param_name_groups[f"vit_layer_{n_layer}"] = {
            "lr_mult": lr_mult,
            "param_names": groups_layers[n_layer],
        }
    lr_mult = lr_layer_decay ** (num_layers + 1)
    logging.info(f"LR at base of tower (input): {lr_mult}")

    param_name_groups["vit_base"] = {
        "lr_mult": lr_mult,
        "param_names": group_base,
    }
    lr_mult = 1.0
    param_name_groups["other"] = {
        "lr_mult": lr_mult,
        "param_names": group_other,
    }
    logging.info(f"LR mult for other parameters, including top of tower (output): {lr_mult}")
    logging.debug(pformat(param_name_groups))
    logging.info(f"{num_parameters} trainable and {num_untrained} frozen weights.")

    return param_name_groups, named_params_dict
