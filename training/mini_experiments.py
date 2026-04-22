"""Named low-resource experiment presets for OpenMythos research."""

from __future__ import annotations


EXPERIMENT_PRESETS: dict[str, dict] = {
    "baseline": {
        "variant": "tiny",
        "dropout": 0.1,
        "dense_ffn_mult": 4.0,
        "attn_type": "gqa",
        "use_moe": False,
        "use_act": False,
    },
    "deep_loops": {
        "variant": "tiny",
        "n_loops": 6,
        "dropout": 0.1,
        "dense_ffn_mult": 4.0,
        "attn_type": "gqa",
        "use_moe": False,
        "use_act": False,
    },
    "act_probe": {
        "variant": "tiny",
        "n_loops": 6,
        "dropout": 0.1,
        "dense_ffn_mult": 4.0,
        "attn_type": "gqa",
        "use_moe": False,
        "use_act": True,
    },
    "moe_probe": {
        "variant": "tiny",
        "n_loops": 4,
        "dropout": 0.1,
        "dense_ffn_mult": 4.0,
        "attn_type": "gqa",
        "use_moe": True,
        "use_act": False,
    },
    "mla_probe": {
        "variant": "tiny",
        "n_loops": 4,
        "dropout": 0.1,
        "dense_ffn_mult": 4.0,
        "attn_type": "mla",
        "use_moe": False,
        "use_act": False,
    },
}
