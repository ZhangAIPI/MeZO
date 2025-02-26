from tqdm import tqdm
from argparse import Namespace
import logging

import torch
from torch.utils.data import DataLoader


from model import PrunableLlamaAttention


logger = logging.getLogger(__name__)


def get_stats(model, calib_loader, args):

    layer_output_x = []
    layer_output_z = []
    layer_wo = []

    for l, layer in enumerate(model.model.layers):
        layer.self_attn = PrunableLlamaAttention(layer.self_attn, r=args.r)
        layer_output_x.append([])
        layer_output_z.append([])
        layer_wo.append(
            [layer.model.o_proj.weight.data.cpu(), layer.model.o_proj.bias.data.cpu()]
        )

    with torch.inference_mode():
        for i, batch in enumerate(
            tqdm(calib_loader, desc="Model forwarding on sample set...")
        ):
            model_inputs = model.prepare_inputs_for_generation(**batch)
            outputs = model(**model_inputs)
            assert outputs is not None
            for l, layer in enumerate(model.model.layers):
                layer_output_x[l].append(layer.cache_X.cpu())
                layer_output_z[l].append(layer.cache_Z.cpu())

    return layer_output_x, layer_output_z, layer_wo
