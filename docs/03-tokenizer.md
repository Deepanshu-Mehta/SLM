# Phase 3 — Train the 16K byte-level BPE tokenizer

**Compute:** 1 CPU container (8 vCPU, 16 GB, `ml_image`) · **Expected cost:** ~$0 · **Status:** DONE

## What this phase does

`modal run modal_app.py::tokenizer`

Trains a **fresh** byte-level BPE tokenizer over every line of
`/data/corpus/*/*.txt` — no pretrained vocabulary is inherited, so the merges are
learned from legal/SEC/web text and the domain's frequent strings ("plaintiff",
"pursuant to", "net revenues") become single tokens.

```
Tokenizer(models.BPE(unk_token="<|unk|>"))
  pre_tokenizer = ByteLevel(add_prefix_space=False)
  decoder       = ByteLevel()
  trainer       = BpeTrainer(vocab_size=16384,
                             special_tokens=[<|bos|>, <|eos|>, <|pad|>, <|unk|>,
                                             <|user|>, <|assistant|>, <|system|>],
                             initial_alphabet=ByteLevel.alphabet())
```

Then wrapped as a `transformers.PreTrainedTokenizerFast` and saved.

### Why these choices

- **Byte-level** → the initial alphabet is all 256 bytes, so **no input can ever
  produce an OOV token**. That matters here: case-law is OCR'd and carries odd
  unicode and mangled characters.
- **16,384 tokens** is deliberately small. At 125M params an embedding table of
  `16384 × 768` is ~12.6M params (~10% of the model); a 32K or 50K vocab would
  spend a much larger share of the budget on embeddings instead of transformer layers.
- **Chat tokens reserved now** (`<|user|>`, `<|assistant|>`, `<|system|>`) even
  though this build has no SFT — adding them later would resize the embedding matrix.

## Output

```
/data/tokenizer/tokenizer.json
/data/tokenizer/tokenizer_config.json
/data/tokenizer/special_tokens_map.json
```

## Validation criteria

| # | Check | Expected |
|---|---|---|
| 1 | vocab size | `vocab_size=16384` |
| 2 | round-trip on a legal sentence | `roundtrip=True` |
| 3 | round-trip on a financial sentence | `roundtrip=True` |
| 4 | tokenizer files on the Volume | 3 files present |

## Recorded gotcha for later (GGUF export, optional in Phase 6)

`llama.cpp`'s `convert_hf_to_gguf.py` rejects a fresh byte-level BPE with
"BPE pre-tokenizer was not recognized", and the tokenizer hash it keys on changes
on every retrain. If we export GGUF, patch the converter to map our unknown
byte-level BPE onto the `gpt-2` pre-tokenizer, and keep the patch idempotent.
Not required for the website deployment.

## Results (actual)

Run: `ap-AJMu657kLgfyh4Eg5Cr1Fe` (detached) · trained over all 670,124 corpus
documents · all 4 checks passed.

| # | Check | Result |
|---|---|---|
| 1 | vocab size | **`vocab_size=16384`** ✓ |
| 2 | legal round-trip | `'The plaintiff shall bear the burden of p...' -> 15 tokens \| roundtrip=True` ✓ |
| 3 | financial round-trip | `'The Company's net revenues increased 12%...' -> 16 tokens \| roundtrip=True` ✓ |
| 4 | files on the Volume | `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json` ✓ |

### The domain specialization is measurable

The legal probe sentence — *"The plaintiff shall bear the burden of proof by a
preponderance of the evidence."* — is 78 characters and encodes to **15 tokens**:
**5.2 characters per token**, against the 4.0 chars/token proxy used for budgeting
in Phases 1–2. The financial probe (86 chars → 16 tokens) gives **5.4 chars/token**.

That is the tokenizer having learned domain-frequent strings ("plaintiff",
"preponderance", "pursuant", "net revenues") as single merges rather than
splitting them into subwords, which is the entire point of training a fresh BPE
on this corpus instead of reusing a general-purpose vocabulary.

It also predicts Phase 4: at >4 chars/token on real text, the true token counts
will come in **below** the chars/4 proxy — the guide's documented ~8% gap
(2.40B proxy → ~2.19B real). That is expected arithmetic, not data loss.
