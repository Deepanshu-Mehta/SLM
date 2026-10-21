"""Phase 8: publish SLM-125M-RAFT (docs/09-raft-release.md §8.2).

    python publish_raft.py card                # assemble .smoke/raft/scale/hf/{README.md, raft_stats.json, raft_decode.py}
    python publish_raft.py upload              # upload that folder to config.HF_RAFT_REPO (public)
    python publish_raft.py verify              # fresh download from the Hub, load, answer two smoke questions

Every number comes from a run artifact under .smoke/raft/ (scores, reports, ledgers), never retyped.
Needs HF_TOKEN (or HUGGINGFACE_TOKEN) in the environment for upload.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import config

ROOT = Path(".smoke/raft")
HF_DIR = ROOT / "scale" / "hf"
J = lambda p: json.loads(Path(p).read_text())


def stats() -> dict:
    head = J(ROOT / "scale" / "headline.json")
    rep = J(ROOT / "scale" / "eval" / "report.json")
    sel = J(ROOT / "scale" / "select_report.json")
    audit = J(ROOT / "scale" / "audit_report.json")
    from raft_gpu_app import SPENT_76_USD, SPENT_78_GPU_USD     # measured container spend (docs/08-raft.md §13)
    spend = {"7.2 Modal CPU": 0.19,                              # E-20 (container lifetimes; no ledger file)
             "7.3": J(ROOT / "trial" / "ledger.json")["spent_usd"], "7.4": J(ROOT / "pilot" / "ledger.json")["spent_usd"],
             "7.6-7.7 GPU": SPENT_76_USD, "7.7 J1": J(ROOT / "pilot-eval" / "ledger.json")["spent_usd"],
             "7.8 data/API": J(ROOT / "scale" / "ledger.json")["spent_usd"], "7.8 GPU": SPENT_78_GPU_USD}
    final = rep["final"]
    run20 = next(r for r in rep["runs"] if r["n"] == final["n"])
    return {
        "test": head["test_copy2"], "test_plain": head["test_plain"], "ceiling": head["ceiling_copy2"],
        "dev_curve": {"2000": J(ROOT / "pilot-eval" / "headline.json")["copy2"]["grounded_accuracy"],
                      **{k.split("_")[1][1:]: v["grounded_accuracy"] for k, v in head.items() if k.startswith("dev_n")}},
        "test_auroc": final["test_decision"]["auroc"], "tau": final["tau"],
        "selected_step": run20["selected_step"], "total_steps": run20["steps"],
        "bpb_delta_pooled": final["bpb_delta_pooled"], "bpb_delta_by_source": final["bpb_delta_by_source"],
        "casehold_pmi_delta_points": final["casehold_pmi_delta_points"],
        "train_rows": sel["train_total"], "train_buckets": sel["train_buckets"], "train_sources": sel["train_sources"],
        "audit": audit["by_bucket"], "spend_usd": {k: round(v, 2) for k, v in spend.items()},
        "spend_total_usd": round(sum(spend.values()), 2),
    }


def card(s: dict) -> str:
    t, tp, c = s["test"], s["test_plain"], s["ceiling"]
    ok = lambda v, op, x: "✅" if (v <= x if op == "<=" else v >= x) else "❌ missed"
    rows = [("Hallucination rate", "hallucination", "<=", 0.15), ("Refusal recall (unanswerable)", "refusal_recall", ">=", 0.80),
            ("Over-refusal (answerable)", "over_refusal_A", "<=", 0.20), ("Grounded accuracy (judged)", "grounded_accuracy", ">=", 0.60),
            ("Citation accuracy", "citation_accuracy", ">=", 0.80), ("Quote fidelity (verbatim)", "quote_fidelity", ">=", 0.90),
            ("Format validity", "format_validity", ">=", 0.99)]
    table = "\n".join(f"| {name} | {op} {x:.2f} | **{t[k]:.3f}** {ok(t[k], op, x)} | {tp[k]:.3f} |" for name, k, op, x in rows)
    curve = " → ".join(f"{int(n) // 1000}K {v:.3f}" for n, v in sorted(s["dev_curve"].items(), key=lambda kv: int(kv[0])))
    b = s["train_buckets"]
    return f"""---
language: en
library_name: transformers
pipeline_tag: text-generation
tags: [llama, legal, rag, raft, grounded-qa, citation, small-language-model]
base_model: {config.HF_REPO}
---

# SLM-125M-RAFT · grounded answers with citations, or a refusal

[`{config.HF_REPO}`](https://huggingface.co/{config.HF_REPO}) (a 125.8M-parameter Llama-style decoder pretrained
from scratch on US case law, SEC filings and educational web text) fine-tuned to **answer only from the passages it
is given**: it names the passage, quotes it verbatim, then answers, or it says
*"{config.RAFT.refusal}"*

- **Website with a live demo and a visual walk-through:** {config.SITE_URL}/grounded
- **Code:** https://github.com/Deepanshu-Mehta/SLM (Phase 7: `docs/08-raft.md`)

## Use it

```python
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
import importlib.util, torch

repo = "{config.HF_RAFT_REPO}"
tok = AutoTokenizer.from_pretrained(repo)
net = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float32).eval()
spec = importlib.util.spec_from_file_location("raft_decode", hf_hub_download(repo, "raft_decode.py"))
rd = importlib.util.module_from_spec(spec); spec.loader.exec_module(rd)

passages = [
    "The Partnership's revenues increased to $4.2 million in 1995 from $3.1 million in 1994, primarily due to higher occupancy at the hotels.",
    "Security for the HM Junior Loan is the Partnership's 100% interest in Marriott DSM LLC. If remaining cash flow is insufficient to pay interest on the HM Junior Loan, interest is deferred and will accrue and compound.",
    "The board of directors met four times during the fiscal year. Each director attended at least 75% of the meetings.",
    "The Company leases its headquarters in Bethesda, Maryland under a lease that expires in 2003.",
]
print(rd.answer(net, tok, passages, "What serves as security for the HM Junior Loan?"))
# answer, s=0.01, passage 2, quote "Security for the HM Junior Loan is the Partnership's 100% interest in Marriott DSM LLC",
#   answer "The Partnership's 100% interest in Marriott DSM LLC"                              <- correct
print(rd.answer(net, tok, passages, "What was the Partnership's net income in 1995?"))
# refuse, s=0.96: "The passages do not contain the answer."                                 <- correct (revenue != net income)
print(rd.answer(net, tok, passages, "How many times did the board of directors meet during the fiscal year?"))
# answer, s=0.23, passage 4, answer "2003"                                                  <- WRONG: the typical failure
```

`raft_decode.py` (in this repo) is the exact decoding used in evaluation:
- **Prompt:** `<|eos|><|user|>Passage 1: …\\nPassage k: …\\nQuestion: …<|assistant|>Source:` (up to 4 passages, about
  160 tokens each; 1,024-token context). Start with `<|eos|>`, never `<|bos|>`.
- **Decision:** the first generated token decides. With s = P(" none") / (P(" none") + P(" Pass")), the model
  **refuses iff s ≥ τ = {s["tau"]}**.
- **Copy-constrained decoding:** after ` Passage k`, the quote is forced to stay a verbatim span of passage k, then
  `\\nAnswer:` follows. Without it the model often paraphrases inside the quote (quote fidelity {tp["quote_fidelity"]:.2f}
  vs {t["quote_fidelity"]:.2f}).

## Results (frozen 1,000-item test set from held-out documents, scored once)

Targets were fixed before any training. "Grounded accuracy" needs a correct answer (judged by `gemini-3.8-flash`
against the reference) **and** the right passage cited.

| Metric | Target | **Copy-constrained (recommended)** | Plain greedy |
|---|---|---|---|
{table}

- **Decision quality:** AUROC {s["test_auroc"]:.3f} (answerable vs unanswerable).
- **Refusal recall by kind:** passage removed {t["refusal_recall_B1"]:.2f}, near-miss question
  {t["refusal_recall_B2"]:.2f}, no or unrelated passages {t["refusal_recall_B3"]:.2f}.
- **Paired consistency:** {t["paired_consistency"]:.3f}. This is answering correctly *with* the passage and
  refusing *without* it, over {t["n_pairs"]} pairs.
- **Learning curve** (judged grounded accuracy on dev): {curve}.
- **Teacher ceiling:** `deepseek-flash` on 200 test items scores grounded {c["grounded_accuracy"]:.2f} and
  hallucination {c["hallucination"]:.2f}. This is inflated by construction, because test items were kept only
  where the teacher's checks agreed.
- **Forgetting:**
  - held-out bits/byte +{s["bpb_delta_pooled"]:.4f} (limit +0.02);
  - CaseHOLD PMI accuracy {s["casehold_pmi_delta_points"]:+.1f} points, reported rather than gated.

**Read this before using it.**
- **What it is good at:**
  - deciding whether the passages contain an answer;
  - citing the right passage ({t["citation_accuracy"]:.0%} of answers);
  - quoting it verbatim;
  - staying in format.
- **What it is not good at:**
  - getting the answer right within the cited passage: about a third are correct;
  - spotting near-miss questions: {t["refusal_recall_B2"]:.0%} refused.
  - It is a 125M model: always check the quote against the passage. It is not legal advice.

## Training

- **Data: {s["train_rows"]:,} examples**, each with 4 passages (the oracle passage plus BM25 and same-document
  distractors):

  | Kind | Rows |
  |---|---|
  | Answerable A | {b["A"]:,} |
  | Oracle removed B1 | {b["B1"]:,} |
  | Near-miss B2 | {b["B2"]:,} |
  | No or unrelated passages B3 | {b["B3"]:,} |
  | Grounded tasks C (summarise, list, yes/no) | {b["C"]:,} |

  Sources: case law {s["train_sources"]["case-law"]:,}, SEC {s["train_sources"]["sec"]:,}, web
  {s["train_sources"]["fineweb-edu"]:,}.
- **How the data was made:**
  - Questions and targets were written by `deepseek-flash` from train-split passages only.
  - Every candidate passed model-free checks (verbatim quotes, no copied question text, no vague references) and
    a blind DeepSeek re-check.
  - Near-miss questions also passed a dedicated "could a careful reader answer this?" check.
  - A 5% `gemini-3.8-flash` audit disagreed on ≤ {max(v["rate"] for v in s["audit"].values()):.1%} per kind.
  - Dev/test come only from held-out documents, scanned for 13-gram overlap with the pretraining corpus.
- **Recipe:**
  - full fine-tune from the fp32 pretraining checkpoint; LR 5e-5 with cosine decay, 3 epochs, batch 64 × 1,024;
  - loss = SFT target loss + 1.0 × replay loss on raw pretraining text (against forgetting);
  - the checkpoint (step {s["selected_step"]} of {s["total_steps"]}) was chosen on a dev proxy under
    copy-constrained decoding, gated on held-out bits/byte;
  - 1 × H100.
- **Cost of the whole phase:** about ${s["spend_total_usd"]:.2f} (teacher APIs plus GPU).

License: not yet chosen; all rights reserved for now (as for the base model).
"""


def cmd_card() -> None:
    s = stats()
    (HF_DIR / "raft_stats.json").write_text(json.dumps(s, indent=2))
    (HF_DIR / "README.md").write_text(card(s))
    shutil.copy("raft_decode.py", HF_DIR / "raft_decode.py")
    print(f"card + stats + raft_decode.py -> {HF_DIR}; spend total ${s['spend_total_usd']}")


def cmd_upload() -> None:
    from huggingface_hub import HfApi

    os.environ.setdefault("HF_TOKEN", os.environ.get("HUGGINGFACE_TOKEN", ""))
    api = HfApi()
    api.create_repo(config.HF_RAFT_REPO, repo_type="model", private=False, exist_ok=True)
    info = api.upload_folder(repo_id=config.HF_RAFT_REPO, folder_path=str(HF_DIR), repo_type="model",
                             commit_message="SLM-125M-RAFT: 20K grounded SFT (Phase 7)")
    print(f"uploaded -> https://huggingface.co/{config.HF_RAFT_REPO} ({info.oid if hasattr(info, 'oid') else info})")


def cmd_verify() -> None:
    import importlib.util
    import tempfile

    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    with tempfile.TemporaryDirectory() as d:
        path = snapshot_download(config.HF_RAFT_REPO, local_dir=d)
        sha = hashlib.sha256(Path(path, "model.safetensors").read_bytes()).hexdigest()
        spec = importlib.util.spec_from_file_location("rd_hub", Path(path) / "raft_decode.py")
        rd = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rd)
        tok = AutoTokenizer.from_pretrained(path)
        net = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32).eval()
        ps = ["The Partnership's revenues increased to $4.2 million in 1995 from $3.1 million in 1994, primarily due to higher occupancy at the hotels.",
              "Security for the HM Junior Loan is the Partnership's 100% interest in Marriott DSM LLC. If remaining cash flow is insufficient to pay interest on the HM Junior Loan, interest is deferred and will accrue and compound.",
              "The board of directors met four times during the fiscal year. Each director attended at least 75% of the meetings.",
              "The Company leases its headquarters in Bethesda, Maryland under a lease that expires in 2003."]
        a = rd.answer(net, tok, ps, "What serves as security for the HM Junior Loan?")
        b = rd.answer(net, tok, ps, "What was the Partnership's net income in 1995?")
        assert a["decision"] == "answer" and a["passage"] == 2 and a["quote"] in ps[1], a      # the card's example 1
        assert b["decision"] == "refuse", b                                                    # the card's example 2
        print(json.dumps({"sha256": sha, "answerable": a, "near_miss": b}, indent=1))


if __name__ == "__main__":
    {"card": cmd_card, "upload": cmd_upload, "verify": cmd_verify}[sys.argv[1]]()
