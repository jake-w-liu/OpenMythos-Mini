from open_mythos.main import (
    ACTHalting,
    Expert,
    GQAttention,
    LoRAAdapter,
    LTIInjection,
    MLAttention,
    MoEFFN,
    MythosConfig,
    OpenMythos,
    RecurrentBlock,
    RMSNorm,
    TransformerBlock,
    apply_rope,
    loop_index_embedding,
    precompute_rope_freqs,
)
from open_mythos.tokenizer import ByteTokenizer, MythosTokenizer
from open_mythos.variants import (
    mythos_1b,
    mythos_1t,
    mythos_3b,
    mythos_10b,
    mythos_50b,
    mythos_100b,
    mythos_500b,
    mythos_nano,
    mythos_small,
    mythos_tiny,
)


def load_tokenizer(model_id: str = "openai/gpt-oss-20b") -> MythosTokenizer:
    return MythosTokenizer(model_id=model_id)


def get_vocab_size(model_id: str = "openai/gpt-oss-20b") -> int:
    return load_tokenizer(model_id).vocab_size

__all__ = [
    "MythosConfig",
    "RMSNorm",
    "GQAttention",
    "MLAttention",
    "Expert",
    "MoEFFN",
    "LoRAAdapter",
    "TransformerBlock",
    "LTIInjection",
    "ACTHalting",
    "RecurrentBlock",
    "OpenMythos",
    "precompute_rope_freqs",
    "apply_rope",
    "loop_index_embedding",
    "mythos_1b",
    "mythos_3b",
    "mythos_10b",
    "mythos_50b",
    "mythos_100b",
    "mythos_500b",
    "mythos_1t",
    "mythos_nano",
    "mythos_tiny",
    "mythos_small",
    "load_tokenizer",
    "get_vocab_size",
    "MythosTokenizer",
    "ByteTokenizer",
]
