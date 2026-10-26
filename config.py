"""Single source of truth for the from-scratch 125M SLM build."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

# Project identity / paths (all paths are relative to the Modal Volume mount).
PROJECT = "slm-125m"
HF_REPO = "Deepanshu027/slm-125m-base"  # Phase 6 push target (the HF account the write token belongs to)
HF_SPACE = "Deepanshu027/slm-125m-api"  # Docker Space target; NOT created (HF needs PRO for Docker Spaces)
HF_RAFT_REPO = "Deepanshu027/slm-125m-raft"  # Phase 8: the grounded SFT model (docs/09-raft-release.md)
API_URL = "https://deepanshu-m23csai--slm-125m-serve-api.us-east.modal.run"  # live API: serve_modal.py (Modal CPU, scale-to-zero)
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
    max_casehold_drop_points: float = 2.0   # reported, not gated, since D-23 (recorded as missed: -4.7)
    # headline targets used to pick tau on dev (§8.1)
    max_over_refusal: float = 0.20
    min_refusal_recall: float = 0.80
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
    # trial 7.3 rule (pre-registered): thinking-off agreement with Gemini on refusal labels was
    # within 2 points of thinking-on (89.5% vs 90.6%), so DeepSeek checks run without thinking
    think_on_refusal_checks: bool = False
    checker_thinking: str = "low"
    # per-request output caps (bound the worst case, §5.5)
    max_tokens_generate: int = 600
    max_tokens_check: int = 250
    max_tokens_check_thinking: int = 2_500


TEACHER = TeacherConfig()

# Approved spend caps per paid stage (USD), docs/08-raft.md §9. Each needs explicit approval.
RAFT_STAGE_CAPS_USD: Mapping[str, float] = {
    "7.3-trial": 3.0,
    "7.4-pilot-data": 4.0,          # lowered from $10 after the trial re-quote (~$1.90, E-22)
    "7.6-pilot-gpu": 4.0,           # also covers the 7.7a retrain (what is left of it)
    "7.7-judge": 1.0,               # Gemini J1 on the pilot candidate (proposed 2026-10-21)
    "7.8-scale-data": 12.0,
    "7.8-full-gpu": 2.0,
}


@dataclass(frozen=True)
class SFTConfig:
    """Training recipe (docs/08-raft.md §7.1)."""

    lr: float = 5e-5                     # D-23 (was 1e-4): LR is the bits/byte lever (E-28, E-29)
    sweep_lrs: tuple[float, ...] = (5e-5, 1e-4, 2e-4)
    sweep_refusal_shares: tuple[float, ...] = (0.15, 0.25, 0.35)
    curve_sizes: tuple[int, ...] = (500, 1_000)
    # 7.6b forgetting follow-up, (lr, replay weight) at the selected 35% refusal share: the selected LR
    # at lambda 0.25 / 0.5 (pre-registered remedy) / 1.0, and the runner-up LR at the remedy weight.
    forgetting_arms: tuple[tuple[float, float], ...] = ((2e-4, 0.25), (2e-4, 0.5), (2e-4, 1.0), (5e-5, 0.5))
    # 7.6c confirmation (E-28): LR is the bits/byte lever, lambda 1.0 the only arm that held CaseHOLD.
    confirm_arm: tuple[float, float] = (5e-5, 1.0)
    pilot_size: int = 2_000
    min_lr_fraction: float = 0.10
    warmup_fraction: float = 0.03
    batch_sequences: int = 64
    micro_batch: int = 32
    epochs: int = 3
    pilot_max_epochs: int = 8
    sft_per_replay: int = 4              # 1 replay window per 4 SFT sequences
    replay_weight: float = 1.0           # lambda: L = L_sft + lambda * L_replay. D-23 (was 0.25)
    keep: str = "select"                 # D-27: best dev proxy score (copy2) among evals passing the bits/byte gate
    full_sizes: tuple[int, ...] = (5_000, 10_000, 20_000)   # 7.8 learning curve (D-27)
    full_evals: tuple[int, ...] = (4, 4, 8)                  # evals per run (20K: every 3/8 epoch)
    tau_pilot: float = 0.75              # 7.7 tau, used for in-training copy2 evals
    refusal_share: float = 0.35          # 7.6 selection; tau trades recall vs over-refusal after training
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    max_new_tokens: int = 128
    gpu: str = "H100"
    gpu_count: int = 1
    seed: int = 1337


SFT = SFTConfig()


# ---- Phase 9: Gemma 2 2B on the same grounded data (docs/10-gemma.md) -----------------
GEMMA_DIR = f"{DATA_ROOT}/gemma"


@dataclass(frozen=True)
class GemmaConfig:
    """LoRA recipe and fixed choices (docs/10-gemma.md §2). Only the LR is swept (D-30: LoRA only)."""

    base: str = "google/gemma-2-2b"
    base_revision: str = "c5ebcd40d208330abc697524c919956e692655cf"       # pinned 2026-10-26 (stored in fp32)
    instruct: str = "google/gemma-2-2b-it"
    instruct_revision: str = "299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8"
    attn: str = "eager"                 # Gemma 2's attention soft-cap is not applied under SDPA (checked in 9.2)
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    pilot_lrs: tuple[float, ...] = (1e-4, 3e-4)
    low_lrs: tuple[float, ...] = (3e-5,)  # 9.4b (D-31): keep the gate; does a lower LR pass it at 5K?
    pilot_size: int = 5_000             # the same select_mix draw as Phase 7's 5K curve point
    pilot_evals: int = 2
    full_size: int = 20_000             # as 7.8 (select_mix(20,000, 0.35) from the 20,340-row pool)
    full_evals: int = 4
    epochs: int = 3
    sft_rows_per_step: int = 52         # Phase 7's SFT rows per step (64 sequences minus 12 replay windows)
    micro_batch: int = 13               # sets the step plan (52 = 13 x 4); actual micro-batches are by tokens
    micro_tokens: int = 9_600           # padded tokens per micro-batch (9.2: 11.4K -> 63.5 GB; 9.4 OOM at 13 longest rows)
    weight_decay: float = 0.0           # on the adapters
    grad_clip: float = 1.0
    gen_batch: int = 32
    max_len: int = 1_024                # longest train row: 985 Gemma tokens (measured); rows are checked
    gpu: str = "H100"                   # all Phase 9 GPU steps (9.3 moved off L40S after the probe)
    seed: int = 1337
    casehold_items: int = 1_200         # seeded subset, before and after (reported metric; 9.2: 7.5 s per 100 items)
    # measured by the 9.2 probe (0 = not yet; 9.4/9.5 refuse to start): seconds per optimizer step at
    # micro_batch (5.88e-5 s/token x 36.0K padded tokens per length-grouped step), and seconds per in-training
    # eval (teacher-forced dev 10 s + held-out bits/byte 28 s + dev copy2 generation <= 145 s; upper bound)
    probe_step_s: float = 2.12
    probe_eval_s: float = 150.0         # 9.4 measured ~150 s per in-training eval (the probe's 183 was an upper bound)


GEMMA = GemmaConfig()

# Spend caps per Phase 9 step, USD (docs/10-gemma.md §5; LoRA-only variant, D-30).
GEMMA_STAGE_CAPS_USD: Mapping[str, float] = {
    "9.2-probe": 0.5,                   # spent $0.15
    "9.3-baselines": 1.0,
    "9.4-pilot": 3.6,                   # re-derived from the 9.2 probe (was 3.0); pending approval
    "9.4b-low-lr": 1.8,                 # D-31: one 5K arm at LR 3e-5 (quoted ≈ $1.3; 9.4 measured ≈ $1.5 per arm)
    "9.5-full": 5.8,                    # re-derived from the 9.2 probe (was 6.0); not run (D-32)
    "9.5-test-5k": 1.6,                 # D-32: test pass on the 5K / LR 3e-5 model (≈ $1.0; raised from 1.4 after C-17)
    "9-judge": 2.0,
}

# ---- Phase 10: RAFT on a 500M model pretrained on the same corpus (docs/11-slm500m-raft.md) ----
SLM500_DIR = f"{DATA_ROOT}/slm500m"


@dataclass(frozen=True)
class Slm500Config:
    """The 500M base (thesreedath/slm-500m-base, Apache-2.0) and the author's 125M control, pinned; the
    RAFT recipe is config.SFT (D-23/D-27) except the LR pilot below."""

    base: str = "thesreedath/slm-500m-base"
    base_revision: str = "4090f98f50ceaa06246649ab5ff38471415d9b9a"
    control: str = "thesreedath/slm-125m-base"          # same corpus and window split as the 500M; licence "other"
    control_revision: str = "15e0afe1d7b78224f79287372c4868baab29df15"
    pilot_lrs: tuple[float, ...] = (5e-5, 2e-5)        # 5e-5 = SLM-125M's choice (D-23); larger models often want less
    pilot_size: int = 5_000
    pilot_evals: int = 4                                # as 7.8's 5K run
    full_size: int = 20_000
    full_evals: int = 8                                 # as 7.8's 20K run
    micro_batch: int = 32                               # 26 SFT + 6 replay, as 7.8; the 10.4 probe may lower it
    replay_tokens: int = 30_000_000                     # re-tokenized train-corpus windows (7.8 drew 14M per 20K run)
    gpu: str = "H100"
    # measured by the 10.4 probe (0 = not yet; 10.5/10.6 refuse to start)
    probe_step_s: float = 0.70          # 10.4: 0.69 s/step at micro 32 (compiled), peak 56 GB
    probe_eval_s: float = 60.0          # 10.4: bpb 7 s + dev copy2 generation ~45 s (base-model upper bound) + teacher-forced 2 s
    # the author's 125M has SLM-125M's geometry: 7.8's measured estimate (raft_gpu_app.full_run: 0.3 s/step, 35 s/eval)
    control_step_s: float = 0.30
    control_eval_s: float = 35.0


SLM500 = Slm500Config()

SLM500_STAGE_CAPS_USD: Mapping[str, float] = {
    "10.4-probe": 0.8,
    "10.5-pilot": 3.0,
    "10.6-full": 4.5,
    "10.7-control": 1.5,
    "10-judge": 1.5,
}

# ---- Phase 11: RAFT on Gemma 3 270M (docs/12-gemma3-270m.md) -------------------------------
GEMMA3_DIR = f"{DATA_ROOT}/gemma3"


@dataclass(frozen=True)
class Gemma3Config:
    """Full fine-tuning of google/gemma-3-270m on the RAFT rows (docs/12-gemma3-270m.md §3)."""

    base: str = "google/gemma-3-270m"
    base_revision: str = "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"     # pinned 2026-10-27 (bf16, 0.54 GB)
    transformers: str = "4.57.6"        # gemma3_text needs >= 4.55; its own image and local venv (.venv-g3)
    attn: str = "eager"                 # Gemma 3 recommends eager for training; no soft-cap, so SDPA would also do
    pilot_lrs: tuple[float, ...] = (2e-5, 5e-5)     # the 500M's and SLM-125M's full fine-tuning LRs (11.3: gate failed)
    lora_lrs: tuple[float, ...] = (1e-4, 3e-5)      # D-34: LoRA as Gemma 2B (config.GEMMA r 32, all linear, embeddings frozen)
    peft: str = "0.17.1"
    lora_step_s: float = 0.50           # conservative: LoRA backward skips the weight gradients
    pilot_size: int = 5_000
    pilot_evals: int = 2                # 11.2: 4 would put the phase 37% over its quote; 2 as Phase 9's pilot
    full_size: int = 20_000
    full_evals: int = 8                 # as SLM-125M's and the 500M's 20K runs
    micro_tokens: int = 32_768          # 11.2: 31.9K padded tokens -> 0.41 s fwd+bwd+opt, peak 53.7 GB
    gpu: str = "H100"
    # measured by the 11.2 probe (0 = not yet; 11.3/11.4 refuse to start)
    probe_step_s: float = 0.50          # ~36K padded tokens per step at 1.27e-5 s/token
    probe_eval_s: float = 60.0          # dev copy2 generation ~45 s + bits/byte ~8 s + teacher-forced ~2 s


GEMMA3 = Gemma3Config()

GEMMA3_STAGE_CAPS_USD: Mapping[str, float] = {
    "11.2-probe": 0.4,
    "11.3-pilot": 2.0,                  # full fine-tuning: spent $0.81, gate failed at both LRs
    "11.3b-lora-pilot": 1.2,            # D-34
    "11.4-full": 2.5,
    "11-judge": 1.0,
}

# ---- Phase 12: RLAIF (docs/13-rlaif.md) ---------------------------------------------------
RLAIF_DIR = f"{DATA_ROOT}/rlaif"


@dataclass(frozen=True)
class RlaifConfig:
    """Sampling, judging and DPO settings (docs/13-rlaif.md §3-§5)."""

    temperature: float = 0.8           # content only: FormatConstraint keeps the syntax; drop to 0.6 if < 98% parse
    min_format_validity: float = 0.98
    headroom_k: int = 8
    betas: tuple[float, ...] = (0.1, 0.3)
    lr: Mapping[str, float] = None      # set below (frozen dataclass default)
    alpha_sft: float = 0.2
    replay_weight_slm: float = 1.0
    pairs_per_step: int = 32
    micro_pairs: int = 8
    epochs: int = 1
    evals: int = 2
    # 12.1 go/no-go (pre-registered): best-of-8 grounded >= greedy + 0.10, or B2 any-refusal >= greedy B2 recall + 0.15
    go_grounded_gain: float = 0.10
    go_b2_gain: float = 0.15
    sample_gpu: str = "L40S"
    train_gpu: str = "H100"


RLAIF = RlaifConfig(lr={"slm125": 2e-6, "gemma3": 1e-6})

RLAIF_STAGE_CAPS_USD: Mapping[str, float] = {
    "12.1-headroom-gpu": 0.6,
    "12.1-judge": 1.2,
    "12.3-sample-gpu": 1.0,
    "12.4-labels": 6.0,
    "12.5-dpo-gpu": 1.5,
    "12.5-judge": 1.0,
}

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
