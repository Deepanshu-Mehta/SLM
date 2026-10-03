# Website plan — SLM-125M on Vercel

**Status:** PLAN (not built yet) · **Hosting cost target:** $0 for the site; inference backend per §9

Two things, one site:

1. **Playground**: type the start of a sentence and watch the model complete it, with
   sample prompts, every sampling control, the model's real numbers, and a
   "how surprising is your sentence?" tool.
2. **How it was built**: a visual, chapter-by-chapter explanation of the data, why it
   was mixed as it was, every pipeline step (streaming, shards, cleaning, dedup,
   MinHash, decontamination, tokenization, packing), the architecture, and training.

Every number on the site comes from run artifacts (`training_stats.json` on the HF model
repo, plus the Phase 1/2/4 reports). Nothing is retyped, so the site cannot drift from
what actually happened.

---

## 1. What the five reference sites do well, and what we take from each

| Site | Strongest idea | We adopt it as |
|---|---|---|
| slm-125m.vercel.app | Metric-card grid; "what this cost to build" table; cold-start note | §3.4 facts strip; Chapter 10 cost card; §5 wake-up state |
| slm125m-live.vercel.app | Blunt base-model disclaimer ("will happily invent citations"); two simple sliders | §3.1 banner; §3.3 *presets* on top of the full controls |
| slm-data-pipeline.vercel.app | Numbered phases in execution order; the "70/20/10 was impossible" reveal; planned vs realized; clickable steps; epoch slider; per-layer parameter counts | The whole "How it was built" spine (§4), Chapters 1, 3, 8, 9 |
| slm-knowledge-capacity.vercel.app | One interactive slider that makes a constraint felt; one-sentence takeaway per section; concrete anchors | Every chapter ends with a **takeaway line**; Chapter 8's capacity slider |
| slm-data-vizuara.vercel.app | (client-rendered; only the title was retrievable) | — |

**Where this site goes further than the references**: live per-token surprise scoring of
your own text, an in-browser tokenizer, a MinHash demo you can edit, a 13-gram
decontamination demo, the train/val leak story (why the split is by document), the
`torch.compile` cost story, and an interactive loss curve that shows the epoch staircase.

---

## 2. Information architecture

```
/                 Playground (landing)           ← most visitors want to try it
/build            How it was built (11 chapters, one long scroll, sticky chapter nav)
/build#data …     deep links to each chapter
/about            Limits, license, links (HF model, code, docs), credits
```

- **Top bar**: name · "base model" badge · Playground · How it was built · HF link.
- **Desktop**: `/build` has a sticky left chapter rail that tracks scroll.
- **Mobile**: the rail becomes a sticky top "Chapter 3 of 11 ▾" dropdown.

---

## 3. Playground (`/`)

### 3.1 Base-model banner (always visible, never dismissible)

> **This is a pretrained base model.** It continues text; it does not answer questions or
> follow instructions. Start a sentence the way a court opinion or a 10-K would, and it
> writes the rest. It is fluent, not factual: it will invent citations, numbers and facts.

### 3.2 Prompt box and sample sentences

- A large textarea, "Start a sentence…", with a character counter (server limit 4,000)
  and a token counter that updates live from the in-browser tokenizer (§7).
- **Sample sentence chips** in three groups. Tapping one fills the box (it replaces,
  and Undo restores the previous text):

| Legal | SEC / financial | General (the model's weak spot, on purpose) |
|---|---|---|
| The court held that the defendant | The Company's net revenues for the fiscal year | Photosynthesis is the process by which |
| The plaintiff alleges that the defendant breached | Item 7. Management's Discussion and Analysis of Financial Condition | The history of the Roman Empire |
| We review the district court's grant of summary judgment de novo | Pursuant to Section 10(b) of the Securities Exchange Act of 1934, | The best way to learn a new language is |
| The Supreme Court of the United States has long recognized that | The Company believes that its existing cash | |
| Accordingly, the judgment of the trial court is | Risk Factors. An investment in our common stock involves | |

The "General" group is deliberate. It lets visitors *see* the limitation instead of
reading about it.

### 3.3 Sampling controls

Three **presets** sit on top: *Precise* (greedy), *Balanced* (default), *Creative*.
An **Advanced** disclosure exposes every knob, each with a one-line plain-English hint
and an ⓘ that explains what it does to the next-token distribution:

| Control | Range | Default | Hint |
|---|---|---|---|
| Temperature | 0 – 1.5 (0 = greedy) | 0.8 | lower = safer and more repetitive, higher = more surprising |
| Top-k | 0 – 200 (0 = off) | 50 | sample only from the k most likely next tokens |
| Top-p | 0.10 – 1.00 | 0.95 | sample from the smallest set covering p of the probability |
| Max new tokens | 16 – 256 | 80 | length of the continuation |
| Min new tokens | 0 – 64 | 16 | stops a base model from ending after one word |
| Repetition penalty | 1.00 – 1.50 | 1.10 | discourages loops |
| Seed | number or 🎲 | random | same seed + settings = same output |

### 3.4 Output

- The prompt is shown in secondary ink, and the completion **streams in** token by token
  in primary ink with a blinking caret. Buttons: **Stop**, Copy, Regenerate.
- Under the output: new tokens · tokens/s · elapsed · "stopped at end-of-document" when
  the model emitted `<|eos|>`.
- **Toggle: "Show tokens"** re-renders the completion as token chips. This teaches what a
  token is, and shows the leading-space convention of byte-level BPE.

### 3.5 Model facts strip (from `/info`, never hard-coded)

Six cards; each has a tooltip that explains the number in one sentence:

| Card | Value |
|---|---|
| Parameters | **125,848,320**, all trainable |
| Held-out loss | **2.127** nats/token |
| Held-out perplexity | **8.39** — "on average, as unsure as choosing among ~8 tokens" |
| Bits per byte | **0.657** — tokenizer-independent; compare it across models |
| Tokens | **8.16B seen** · 2.04B unique · 4 epochs |
| Context · vocab | 1,024 tokens · 16,384 BPE |

The strip ends with: *"Pretrained from scratch on US case law, SEC filings and educational
web text — [how it was built →]"*.

### 3.6 "How surprising is your sentence?" (uses `/score`)

- Paste any text and the model scores every token.
- Tokens render as chips **shaded by surprise** (a sequential single-hue ramp,
  light→dark), with a legend and a hover readout: *"' plaintiff': p = 0.0008 (7.2 nats)"*.
- A headline compares your text's perplexity and bits/byte with the model's held-out
  average.
- Two buttons, **"Try legal"** and **"Try nonsense"**, reproduce the measured contrast:
  perplexity **5.86 vs 488**.

### 3.7 Layout and responsiveness

- **≥1024px**: two columns. Left: prompt, chips, presets, controls. Right: output, sticky.
  Facts strip full-width below.
- **640–1023px**: one column, with controls in a collapsible panel.
- **<640px**: one column; chips scroll horizontally; controls in a **bottom sheet**
  ("⚙ Sampling"); a sticky Generate button at the bottom of the viewport; tap
  targets ≥ 44px; the output box grows with content.
- Tested at 360, 390, 768, 1024 and 1440px; no horizontal page scroll at any width.

---

## 4. How it was built (`/build`): 11 chapters

Each chapter has the same rhythm: **one visual → what we did → why → the real number →
a one-line takeaway**. Each chart has a "view as table" toggle.

| # | Chapter | Visual (all charts: hover tooltips, legend + direct labels, table view) | The "why" it teaches | Source of numbers |
|---|---|---|---|---|
| 0 | **Hero** | Five big numbers that count up: 13.6B tokens available → 2.04B kept → 8.16B seen → 125.8M params → ≤ $25 | — | stats + reports |
| 1 | **The data budget problem** | *Supply vs demand* grouped bars: measured clean supply (case-law 0.81B, SEC 1.16B, fineweb 11.67B) next to what a 70/20/10 mix of 10B would demand (7B / 2B / 1B). Case-law's bar falls visibly short | 70/20/10 was **impossible**: the legal sources only hold ~2B tokens. So we take all of them and cap the web | Phase 0 measure |
| 1b | **The mix we got** | One 100% stacked bar: case-law 35.1 · SEC 42.2 · fineweb 22.8. Hover shows tokens and docs. Toggle "planned budget vs realized" | Legal-first: 77% legal. SEC came in under budget because the source ran out | index.json |
| 2 | **Streaming, not downloading** | Animated diagram: HF parquet shards (10 · 5 · 14→5) fan out to **20 worker tiles**. Each tile fills to its token cap and shows its real streamed/kept counts. A "preemption" button kills one tile and only that tile restarts | Never materialise terabytes; one container per shard keeps a preemption's blast radius to one shard (this happened twice in Phase 4) | per-shard Phase 1 log |
| 3 | **Cleaning** | (a) A **funnel**: 718,780 streamed → −20,131 too short → −685 OCR garble → −6 non-English → 697,958 kept. (b) A **before/after document card**: a real scanned opinion, with each of the 6 steps as a toggle that strikes through what it removes (short lines, `Page 3 of 9`, `/s/` signatures, garbled OCR) | Cheapest check first; every drop counted by reason. The OCR gate exists because case-law is scanned (one shard held 578 of the 685 garbled documents) | phase1_report + real excerpts |
| 4 | **Duplicates & MinHash** | Editable **MinHash demo**: two court-order snippets, their 5-word shingles highlighted where they overlap, live estimated vs true Jaccard, and an LSH-bucket animation. Then counts: 1,606 near-dups (Jaccard ≥ 0.8) · 1,989 SEC exact dups · 62 web exact dups | Repeated text wastes budget and teaches memorization; MinHash finds near-copies in one pass instead of comparing every pair | phase2_report |
| 5 | **Decontamination** | A **13-gram sliding window** sweeps a sample document; when a window matches the benchmark it lights up and the whole document is dropped. Bar: 24,002 case-law (11.6%) vs 175 SEC docs removed | If test questions leak into training, the evaluation lies. CaseHOLD is built from court opinions, which is why case-law lost 11.6% | phase2_report |
| 6 | **The tokenizer** | **Live tokenizer** in the browser: type, see tokens and ids. A bar of chars/token by source (case-law 4.13 · SEC 4.98 · web 4.23) | A fresh 16K byte-level BPE learns "plaintiff" and "pursuant" as single tokens. Byte-level means no unknown tokens, ever. 16K keeps the embedding table at 10% of the parameters | stats + tokenizer.json |
| 7 | **Packing & the honest split** | Documents as coloured bars flowing into fixed 1,024-token windows, separated by `<\|eos\|>`. A toggle "split by window ↔ split by document" shows an SEC filing's 18.8 windows scattering across train/val in the first mode and staying whole in the second | Why we rebuilt Phase 4: a window split lets 18 sibling windows of each SEC val filing into training. Over 4 epochs that measures recall, not generalisation | index.json + docs |
| 8 | **Architecture** | Interactive **block diagram**: embeddings (12.58M) → 12 × [RMSNorm → attention 2.36M → RMSNorm → SwiGLU MLP 7.08M] → final RMSNorm → tied LM head. Click a block for shapes and parameter count. Beside it, a **parameter-share bar**: MLP 67.5% · attention 22.5% · embeddings 10.0% · norms 0.02%. A **capacity slider** (knowledge-capacity idea) to explore what scaling layers/width would cost in parameters | Why Llama-style (RoPE, SwiGLU, RMSNorm, tied embeddings), and why a 16K vocabulary leaves room for depth | config |
| 9 | **Training** | **Interactive loss curve**: train vs held-out over 8.16B tokens, with epoch boundaries and the LR schedule as a small panel beneath (never a dual axis). An **epoch scrubber** shows held-out bits/byte per epoch (0.750 → 0.697 → 0.670 → 0.657, gains halving). A bar compares **eager 15.2% vs compiled 32.4% MFU → $51.67 vs $24.28** | Repetition helped through all 4 epochs (val improved at every eval) but with halving returns; `torch.compile` decided whether 4 epochs fit the budget | training_stats.json |
| 10 | **Results, limits, cost** | Bits/byte by source (case-law ≈0.69 vs web ≈1.02); three real samples (legal ✓, SEC ✓, photosynthesis ✗ fluent but false); cost table: data $0 · calibration ≤ $0.54 · training ≤ $24.78 | It speaks the domain; it does not know facts; why the honest metric is held-out bits/byte, not MMLU | stats + docs |

**Data plumbing for `/build`:** a small script (`site/scripts/build-data.ts`) runs at build
time. It pulls `training_stats.json` from the HF repo and merges the Phase 1/2/4 reports
plus the per-shard Phase 1 log into `site/data/pipeline.json`. Charts import that file:
no runtime API calls, so `/build` works even when the inference backend is asleep.

**Real text excerpts** (Chapters 3–5): picked by a script from the public datasets
(raw → cleaned pairs) and stored in `pipeline.json` with their source ids, so every
before/after example is genuine.

---

## 5. Inference client and the "waking up" state

The site talks to one interface, `InferenceClient { info(); generate(req, onToken);
score(text) }`, with one implementation per backend option in §9. Swapping backends is
one environment variable; the UI never changes.

Backend states shown in a status pill: **Ready**, **Waking the model…** (with an elapsed
counter), **Generating**, **Error** (with a retry button).
- On page load the site pings `/health` in the background, so the model is often awake
  by the time the visitor presses Generate.
- On a cold backend: retry with backoff, show "first request after idle can take
  ~20–40 s", and keep the visitor's prompt.

---

## 6. Visual design system

- **Palette**: the dataviz reference palette. Categorical order is fixed per *entity*:
  case-law = slot 1 blue `#2a78d6`, SEC = slot 2 orange `#eb6834`, web = slot 3 aqua
  `#1baf7a`, the same everywhere on the site. Train/val curves reuse slots 1/2
  (validated: CVD ΔE 24.7). Surprise heat = one sequential hue.
- **Validated 2026-10-03** with the palette validator: the 3-source set passes every check
  in light and dark mode (worst colour-blind ΔE 9.2). One light-mode WARN: aqua is 2.74:1
  on the light surface, so wherever aqua appears it carries a visible direct label
  (already mandatory here, along with a table view).
- **Light and dark** themes are both designed (dark stepped from the same ramps, not
  inverted). Text always uses ink tokens, never series colours.
- **Marks**: 2px lines, thin bars with 4px rounded ends, hairline grid, one y-axis per
  chart, no pies except the 100% bar.
- **Type**: Inter for UI, JetBrains Mono for tokens/code, tabular numerals for every metric.
- **Motion**: scroll-triggered, under 600ms, and fully disabled under
  `prefers-reduced-motion`.

---

## 7. Tech stack

| Concern | Choice | Why |
|---|---|---|
| Framework | **Next.js (App Router) + TypeScript** | Vercel-native; static pages for `/build`, client components for the playground |
| Styling | **Tailwind CSS** + CSS variables for theme tokens | Fast responsive work; light/dark via tokens |
| Standard charts | **Observable Plot** | Concise, good defaults for line/bar charts, easy table views |
| Custom diagrams | **D3 + hand-written SVG** (shards, funnel, MinHash, sliding window, packing, architecture) | These are explanatory diagrams, not charts; no library draws them well |
| Animation | **Framer Motion** (scroll-triggered) | Declarative, respects reduced motion |
| In-browser tokenizer | **`@huggingface/transformers` `AutoTokenizer`** loading our `tokenizer.json` (1.1 MB) from the HF repo | Live token counts and the Chapter 6 demo with zero backend calls |
| Streaming | `fetch` + ReadableStream over SSE from `/generate/stream` | Works on mobile Safari; no extra dependency |
| Hosting | **Vercel** (Hobby, free), GitHub-connected, preview deploy per PR | |
| Quality gates | Lighthouse ≥ 90 (perf/a11y/best-practices) on mobile; axe a11y check; Playwright smoke tests at 390px and 1440px | |

---

## 8. Build milestones

| # | Milestone | Done when |
|---|---|---|
| 1 | Scaffold, theme tokens, layout shell, nav, `pipeline.json` build script | Builds on Vercel preview; numbers render from data, not literals |
| 2 | Playground: prompt, chips, presets and controls, streaming output, facts strip, wake-up state | Completes a prompt end-to-end on desktop **and** a 390px phone viewport |
| 3 | Surprise tool and token-chip view | Legal vs nonsense reproduces ppl 5.86 vs 488 |
| 4 | `/build` Chapters 0–3 (budget, mix, streaming, cleaning) | Every chart has hover, legend/labels and a table view |
| 5 | Chapters 4–7 (MinHash, decontamination, tokenizer, packing) | Interactive demos work on touch |
| 6 | Chapters 8–10 (architecture, training, results/cost) | Loss curve matches `training_stats.json` exactly |
| 7 | Polish: dark mode, reduced motion, Lighthouse, Playwright | Quality gates in §7 pass |

---

## 9. Decision needed: where inference runs

HF now requires a **PRO subscription** to host Docker/Gradio Spaces, even on free CPU
(`402 Payment Required` when creating `Deepanshu027/slm-125m-api`). The FastAPI server in
`serve/` is written and tested locally either way.

| Option | Cost | UX | Work |
|---|---|---|---|
| **A. Modal CPU endpoint, scale-to-zero** (`serve/` as a Modal `asgi_app`, `max_containers=1`, short idle timeout, per-IP rate limit) | **$0 when idle**; about $0.13 per *hour of active use*; worst case if abused is capped at one container | Fast once warm; ~10–20 s cold start | Small: reuse the tested server |
| **B. In-browser inference** (ONNX export, int8 ~130 MB, run with transformers.js WebGPU/WASM; weights served free from the HF repo) | **$0 forever**; nothing to switch off; no abuse risk | ~130 MB first load (then cached); slower on older phones | Larger: ONNX export + validation |
| **C. HF PRO** ($9/month) and the Docker Space as designed | $9/month fixed | Same as A | None: `publish_hf.py space` is ready |

## 10. Open questions

1. Backend option A, B or C (§9).
2. License for the model and site (the model card currently leaves it unset).
3. Custom domain, or the default `*.vercel.app`?
4. Should the site name a person or organization in its credits?
