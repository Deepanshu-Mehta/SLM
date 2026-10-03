"""Model factory for the 125M SLM. Geometry comes from config.MODEL only."""

from __future__ import annotations

from typing import TYPE_CHECKING

import config

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from transformers import LlamaConfig, LlamaForCausalLM

# Token ids are fixed by the Phase 3 tokenizer's special-token ordering
# (config.SPECIAL_TOKENS insertion order), verified against tokenizer.json:
#   0 <|bos|>   1 <|eos|>   2 <|pad|>   3 <|unk|>
BOS_TOKEN_ID = 0
EOS_TOKEN_ID = 1
PAD_TOKEN_ID = 2
UNK_TOKEN_ID = 3

# Prompts start with EOS, NOT BOS. The packed training stream is
#   doc, <|eos|>, doc, <|eos|>, ...
# with no <|bos|> anywhere: Phase 4 encodes with add_special_tokens=False and only
# appends EOS, and the tokenizer has no post-processor. So every document the model
# saw began immediately after an EOS, while id 0 was never once an input token and
# its input embedding is untrained. config.json still records bos_token_id=0 because
# that is the true vocabulary entry; inference code must prepend this id instead.
PROMPT_PREFIX_TOKEN_ID = EOS_TOKEN_ID


def build_llama_config(attn_implementation: str = "sdpa") -> "LlamaConfig":
    """A LlamaConfig built strictly from config.MODEL, with explicit token ids."""
    from transformers import LlamaConfig

    return LlamaConfig(
        **config.MODEL.to_llama_kwargs(),
        bos_token_id=BOS_TOKEN_ID,
        eos_token_id=EOS_TOKEN_ID,
        pad_token_id=PAD_TOKEN_ID,
        attn_implementation=attn_implementation,
    )


def build_model(attn_implementation: str = "sdpa") -> "LlamaForCausalLM":
    """Fresh, randomly-initialised model in fp32 (master weights live in fp32)."""
    from transformers import LlamaForCausalLM

    model = LlamaForCausalLM(build_llama_config(attn_implementation))
    if config.MODEL.tie_word_embeddings:
        model.tie_weights()
    return model


def count_parameters(model: "LlamaForCausalLM") -> dict[str, int]:
    """Exact parameter counts. Tied weights are counted once."""
    total = sum(p.numel() for p in model.parameters())
    embed = model.model.embed_tokens.weight.numel()
    return {"total": total, "embedding": embed, "non_embedding": total - embed}
