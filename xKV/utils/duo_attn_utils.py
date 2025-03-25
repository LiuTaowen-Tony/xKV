import numpy as np
import os
import json


def sparsify_attention_heads(full_attention_heads, threshold=None, sparsity=None):
    # add a very small random noise to full_attention_heads to break ties
    full_attention_heads += np.random.uniform(0, 1e-6, full_attention_heads.shape)
    if sparsity is not None:
        # ignore the threshold and use the sparsity
        # set the sparsity small values to 0 and others to 1
        threshold = np.quantile(full_attention_heads, sparsity)
    else:
        assert threshold is not None, "Either threshold or sparsity must be provided"

    if sparsity >= 1:
        # all heads are pruned
        threshold = 2
    if sparsity <= 0:
        # no heads are pruned
        threshold = -1

    full_attention_heads = (full_attention_heads >= threshold).astype(float)
    sparsity = 1 - np.mean(full_attention_heads)
    return full_attention_heads, sparsity


def load_attn_pattern(attn_load_dir):
    full_attention_heads = np.loadtxt(
        os.path.join(attn_load_dir, "full_attention_heads.tsv"),
        dtype=float,
        delimiter="\t",
    )
    full_attention_heads = np.clip(full_attention_heads, 0, 1)
    config = json.load(open(os.path.join(attn_load_dir, "config.json")))
    sink_size = config["sink_size"]
    recent_size = config["recent_size"]
    return full_attention_heads, sink_size, recent_size