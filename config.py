"""Single source of truth for the from-scratch 125M SLM build."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

# Project identity / paths (all paths are relative to the Modal Volume mount).
PROJECT = "slm-125m"
HF_REPO = "Deepanshu027/slm-125m-base"  # Phase 6 push target (the HF account the write token belongs to)
HF_SPACE = "Deepanshu027/slm-125m-api"  # Docker Space target; NOT created (HF needs PRO for Docker Spaces)
API_URL = "https://mdeepanshu2706--slm-125m-serve-api.modal.run"  # live API: serve_modal.py (Modal CPU, scale-to-zero)
SITE_URL = "https://slm-125m-legal-sandy.vercel.app"  # website (Vercel)

VOLUME_NAME = "slm-125m"
DATA_ROOT = "/data"
CLEAN_DIR = f"{DATA_ROOT}/clean"          # Phase 1
CORPUS_DIR = f"{DATA_ROOT}/corpus"        # Phase 2
TOKENIZER_DIR = f"{DATA_ROOT}/tokenizer"  # Phase 3
TOKENS_DIR = f"{DATA_ROOT}/tokens"        # Phase 4
TRAIN_TOKENS_DIR = f"{TOKENS_DIR}/train"
VAL_TOKENS_DIR = f"{TOKENS_DIR}/val"
CKPT_DIR = f"{DATA_ROOT}/checkpoints"     # Phase 5 (not in this brief)
BASE_CKPT_DIR = f"{CKPT_DIR}/base"
RESUME_CKPT_PATH = f"{CKPT_DIR}/ckpt.pt"
METRICS_PATH = f"{CKPT_DIR}/metrics.jsonl"

HF_SECRET_NAME = "huggingface-token"


@dataclass(frozen=True)
class ModelConfig:
    """Maps 1:1 to transformers.LlamaConfig. ~125.8M params with tied embeddings."""

    vocab_size: int = 16_384
    hidden_size: int = 768
    intermediate_size: int = 3_072        # SwiGLU inner
    num_hidden_layers: int = 12
    num_attention_heads: int = 12         # head dim 64
    num_key_value_heads: int = 12         # == heads -> MHA
    max_position_embeddings: int = 1_024  # context length
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-5
    hidden_act: str = "silu"              # SwiGLU
    tie_word_embeddings: bool = True
    attention_bias: bool = False

    def to_llama_kwargs(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "max_position_embeddings": self.max_position_embeddings,
            "rope_theta": self.rope_theta,
            "rms_norm_eps": self.rms_norm_eps,
            "hidden_act": self.hidden_act,
            "tie_word_embeddings": self.tie_word_embeddings,
            "attention_bias": self.attention_bias,
        }

    def approx_params(self) -> int:
        e = self.vocab_size * self.hidden_size
        h, i = self.hidden_size, self.intermediate_size
        kv = self.num_key_value_heads * (h // self.num_attention_heads)
        attn = h * h + 2 * (h * kv) + h * h
        mlp = 3 * h * i
        per_layer = attn + mlp + 2 * h
        final_norm = h  # model.norm (final RMSNorm), one weight per hidden dim
        return e + self.num_hidden_layers * per_layer + final_norm


MODEL = ModelConfig()

SPECIAL_TOKENS: Mapping[str, str] = {
    "bos_token": "<|bos|>",
    "eos_token": "<|eos|>",
    "pad_token": "<|pad|>",
    "unk_token": "<|unk|>",
}
EXTRA_CHAT_TOKENS: tuple[str, ...] = ("<|user|>", "<|assistant|>", "<|system|>")


@dataclass(frozen=True)
class Source:
    name: str
    hf_id: str
    token_budget: int      # stop streaming this source at ~this many clean tokens
    text_field: str
    split: str = "train"
    config_name: str | None = None
    strict_ocr: bool = False


# Legal-first mix (Choice A). Legal sources cap ~2B tokens, so take all of them
# and cap web at 0.5B. NOT 70/20/10. Realized ~40/40/20 (case-law/sec/web).
DATA_MIX: tuple[Source, ...] = (
    Source("case-law", "HFforLegal/case-law", 1_000_000_000, "document",
           split="us", strict_ocr=True),
    Source("sec", "PleIAs/SEC", 1_300_000_000, "text", split="train"),
    Source("fineweb-edu", "HuggingFaceFW/fineweb-edu", 500_000_000, "text",
           split="train", config_name="sample-10BT"),
)

TARGET_TOKENS: int = 2_500_000_000  # Chinchilla REFERENCE only (~20 tok/param).
# The real training length is PRETRAIN_EPOCHS x the train tokens recorded in
# /data/tokens/index.json -- see pretrain_steps(). Do not train off this number.
CHARS_PER_TOKEN: float = 4.0

# Held OUT of training; Phase 2 strips docs resembling these.
EVAL_HOLDOUT: tuple[str, ...] = ("coastalcph/lex_glue", "casehold/casehold")


@dataclass(frozen=True)
class CleanConfig:
    min_line_chars: int = 40
    max_nonalnum_ratio: float = 0.30
    min_doc_chars: int = 600
    repetition_top_k: int = 10
    max_repetition_ratio: float = 0.50
    ngram_n: int = 4
    lang_sample_chars: int = 5_000
    nonword_ratio_max: float = 0.20     # OCR gate: drop if >20% words non-dictionary
    ocr_min_tokens: int = 50
    dict_path: str = "/usr/share/dict/words"


CLEAN = CleanConfig()

SEQ_LEN: int = 1_024
VAL_ONE_IN_N_DOCS: int = 100            # a document goes wholly to val when
                                        # blake2b(text) % 100 == 0 (99/1 by document)
TOKENS_DTYPE: str = "uint16"


@dataclass(frozen=True)
class TrainConfig:
    seq_len: int = SEQ_LEN
    micro_batch_size: int = 32
    global_batch_tokens: int = 524_288
    lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_tokens: int = 200_000_000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    ckpt_every_steps: int = 500
    log_every_steps: int = 20
    eval_every_steps: int = 1_000
    seed: int = 1337


TRAIN = TrainConfig()

PRETRAIN_GPU = "H100"
PRETRAIN_GPU_COUNT = 8
PRETRAIN_EPOCHS = 4               # passes over the fixed corpus (~8.16B tokens seen)
BUDGET_CAP_USD = 50.0

# Modal on-demand GPU rates, USD per GPU-second. Verified against modal.com/pricing
# on 2026-09-22. Used by the trainer's --max-usd guard to project spend live.
GPU_RATE_PER_SEC: Mapping[str, float] = {
    "H100": 0.001097,   # $3.95 / GPU-hour (H100 SXM5)
    "H200": 0.001261,   # $4.54 / GPU-hour
    "A100-80GB": 0.000694,
    "A100-40GB": 0.000583,
    "L40S": 0.000542,
    "A10": 0.000306,
    "T4": 0.000164,
}
# Modal bills requested CPU and memory on top of the GPU rate; the guard inflates
# the GPU-only projection by this factor so the cap is not breached by overhead.
COST_OVERHEAD_FACTOR: float = 1.15

# Dense bf16 tensor-core peak per GPU, FLOP/s. MFU = 6 * params * tokens/s / (n_gpu * peak).
GPU_PEAK_BF16_FLOPS: Mapping[str, float] = {"H100": 989e12, "H200": 989e12, "A100-80GB": 312e12}

# UTF-8 text bytes per token over the v2 (document-level) val split, measured by
# verify_model.py with skip_special_tokens=True. Bits-per-byte = loss_nats / (ln2 * this).
# Re-measure whenever the val split is rebuilt.
VAL_BYTES_PER_TOKEN: float = 4.669241


def pretrain_steps(train_tokens: int, epochs: int = PRETRAIN_EPOCHS) -> int:
    """Optimizer steps for the full run. train_tokens comes from tokens/index.json."""
    return (train_tokens * epochs) // TRAIN.global_batch_tokens


def projected_cost_usd(gpu_seconds_per_gpu: float, gpu: str = PRETRAIN_GPU,
                       gpu_count: int = PRETRAIN_GPU_COUNT) -> float:
    """Spend for a wall-clock duration, including Modal's CPU/memory overhead."""
    return gpu_seconds_per_gpu * gpu_count * GPU_RATE_PER_SEC[gpu] * COST_OVERHEAD_FACTOR

# ---- Phase 7: grounded SFT / RAFT (docs/08-raft.md) --------------------------------
RAFT_DIR = f"{DATA_ROOT}/raft"
RAFT_INIT_CKPT = RESUME_CKPT_PATH        # fp32 Phase 5 weights; load state["model"] only


@dataclass(frozen=True)
class RaftConfig:
    """Data recipe. Ratios are shares of the training examples (docs/08-raft.md §3)."""

    k_passages: int = 4
    passage_min_tokens: int = 60
    passage_max_tokens: int = 160
    question_max_tokens: int = 48
    target_max_tokens: int = 128
    refusal: str = "The passages do not contain the answer."
    # buckets: answerable / oracle removed (paired with A) / near miss / no context / tasks
    mix: tuple[tuple[str, float], ...] = (
        ("A", 0.55), ("B1", 0.15), ("B2", 0.075), ("B3", 0.025), ("C", 0.20))
    source_mix: tuple[tuple[str, float], ...] = (
        ("case-law", 0.40), ("sec", 0.40), ("fineweb-edu", 0.20))
    train_examples: int = 10_000
    pilot_train_examples: int = 2_400    # enough for 15/25/35% refusal mixes of 2,000
    dev_examples: int = 500
    test_examples: int = 1_000
    dev_hash_mod: int = 3                # blake2b(text + "|dev") % 3 == 0 -> dev, else test
    # distractors (§4.1 step 4)
    distractor_pool_per_source: int = 250_000
    same_doc_distractors_max: int = 2
    bm25_sample_top: int = 20
    oracle_jaccard_max: float = 0.5      # word 5-gram Jaccard vs the oracle
    # dev/test leakage scan against the whole pretraining train corpus (§1)
    leak_ngram: int = 13
    leak_max_fraction: float = 0.10
    # forgetting guards (§8.2)
    eval_windows_per_source: int = 200
    max_bpb_increase: float = 0.02
    max_casehold_drop_points: float = 2.0
    seed: int = 1337


RAFT = RaftConfig()


@dataclass(frozen=True)
class TeacherConfig:
    """Teacher / checker models and prices (USD per 1M tokens), checked 2026-10-06 to 10-15."""

    writer: str = "deepseek-flash"              # DeepSeek-V4.1-Flash; OpenAI-compatible API
    writer_base_url: str = "https://api.deepseek.com"
    checker: str = "gemini-3.8-flash"           # Batch API
    # DeepSeek off-peak prices; peak (below) is double
    writer_in: float = 0.15
    writer_cached_in: float = 0.003
    writer_out: float = 0.60
    writer_peak_multiplier: float = 2.0
    # DeepSeek peak hours, UTC, Monday-Friday: [start, end)
    peak_hours_utc: tuple[tuple[int, int], ...] = ((1, 4), (6, 10))
    checker_batch_in: float = 0.375
    checker_batch_out: float = 1.875
    # thinking: DeepSeek effort when a check thinks (refusal labels, dev/test); Gemini level
    check_thinking_effort: str = "low"
    checker_thinking: str = "low"
    # per-request output caps (bound the worst case, §5.5)
    max_tokens_generate: int = 600
    max_tokens_check: int = 250
    max_tokens_check_thinking: int = 2_500


TEACHER = TeacherConfig()

# Approved spend caps per paid stage (USD), docs/08-raft.md §9. Each needs explicit approval.
RAFT_STAGE_CAPS_USD: Mapping[str, float] = {
    "7.3-trial": 3.0, "7.4-pilot-data": 10.0, "7.6-pilot-gpu": 4.0,
    "7.8-scale-data": 12.0, "7.8-full-gpu": 2.0,
}


@dataclass(frozen=True)
class SFTConfig:
    """Training recipe (docs/08-raft.md §7.1)."""

    lr: float = 1e-4
    sweep_lrs: tuple[float, ...] = (5e-5, 1e-4, 2e-4)
    sweep_refusal_shares: tuple[float, ...] = (0.15, 0.25, 0.35)
    curve_sizes: tuple[int, ...] = (500, 1_000)
    pilot_size: int = 2_000
    min_lr_fraction: float = 0.10
    warmup_fraction: float = 0.03
    batch_sequences: int = 64
    micro_batch: int = 32
    epochs: int = 3
    pilot_max_epochs: int = 8
    sft_per_replay: int = 4              # 1 replay window per 4 SFT sequences
    replay_weight: float = 0.25          # lambda: L = L_sft + lambda * L_replay
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    max_new_tokens: int = 128
    gpu: str = "H100"
    gpu_count: int = 1
    seed: int = 1337


SFT = SFTConfig()

STAGES: tuple[str, ...] = (
    "setup", "clean", "dedup", "tokenizer", "tokenize", "pretrain", "deploy", "sft",
)


if __name__ == "__main__":
    p = MODEL.approx_params()
    print(f"{PROJECT}")
    print(f"model: {p:,} params (~{p/1e6:.1f}M) | vocab {MODEL.vocab_size} | "
          f"{MODEL.num_hidden_layers}L/{MODEL.hidden_size}d/"
          f"{MODEL.num_attention_heads}h kv={MODEL.num_key_value_heads}")
    print(f"target tokens: {TARGET_TOKENS/1e9:.1f}B (~{TARGET_TOKENS/p:.0f} tok/param)")
    print(f"stages: {' -> '.join(STAGES)}")
