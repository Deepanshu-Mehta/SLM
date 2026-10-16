"""Teacher and checker prompts (docs/08-raft.md §6), split into a fixed system part and a
variable user part. The fixed part comes first in every request, so DeepSeek serves it from
its prefix cache. Every prompt asks for JSON and shows its shape (DeepSeek JSON mode
requires the word "json" and an example).
"""

from __future__ import annotations

SOURCE_DESC = {"case-law": "US court opinion", "sec": "SEC filing",
               "fineweb-edu": "educational web page"}
TASK_TYPES = ("summarize", "extract_list", "yes_no_check")

# ----------------------------------------------------------------------- generators
G1_SYSTEM = """You write training data for a small model that must answer ONLY from passages it is given.
You receive one passage from a {source_desc}. Write up to 2 question-answer pairs that this
passage alone answers.

Rules:
1. The answer must be stated explicitly in the passage, in so many words. No outside knowledge
   and no inference (the checker rejects anything implied).
2. Each question must be self-contained and specific: name the party, company, court, statute,
   period or amount it concerns, using names that appear in the passage. Someone reading a
   different passage on a similar topic must not be able to answer it.
3. Paraphrase. Apart from names, never copy more than 5 consecutive words of the passage into
   the question.
4. "quote" is the shortest exact span of the passage (at most 40 words, copied character for
   character) that contains the answer.
5. "answer" is at most 30 words, direct, and consistent with the quote. Copy numbers, dates and
   names exactly as written.
6. Vary "type" across: fact, number_or_date, party_or_entity, rule_or_holding, definition, yes_no.
   A yes_no answer may be "No" when the passage states the opposite.
7. If the passage is boilerplate, a table fragment, a list of citations, garbled OCR, or lacks
   the context for a specific question, return {{"items": []}}.

Reply with JSON only: {{"items": [{{"question": "...", "answer": "...", "quote": "...", "type": "..."}}]}}"""

G2_SYSTEM = """You write test questions for a model that must refuse when its passages lack the answer.
You receive one passage from a {source_desc}. Write ONE question that a reader would expect this
passage to answer (same parties, subject and time period) but that it does NOT answer: it asks
for a specific detail the passage never states. The question must not be answerable by inference
from the passage, and must not be about an obviously unrelated topic. Name the entity it concerns.

Reply with JSON only: {{"question": "...", "missing_detail": "..."}}, or {{"question": null}} if no
such question is possible."""

G3_SYSTEM = """You write instructions for a model that must work ONLY from passages it is given.
Write ONE instruction of type {task} that can be completed using only this passage, and its ideal
response. Refer to the passage only as "the passage" (it will be replaced with its number).
- summarize: the instruction asks for a one-sentence summary; response at most 35 words, with no
  fact that is not in the passage.
- extract_list: the instruction asks for all parties, dates, amounts, or defined terms of one
  kind; response lists them exactly as written, separated by "; ".
- yes_no_check: the instruction asks whether the passage states a specific claim; response is
  "Yes." or "No." then a verbatim quote of at most 30 words that decides it. Make the claim false
  ("No.") about half of the time, by changing one detail of something the passage does state.

Reply with JSON only: {{"instruction": "...", "response": "...", "quote": "..." or null}}"""

GEN_USER = 'Source: {source_desc}\nPassage:\n"""{passage}"""'

# -------------------------------------------------------------------------- checkers
V1_SYSTEM = """You check training examples for a model that must answer only from the passages it is given.
Using ONLY the passages below (no outside knowledge), decide:
- answerable: true only if at least one passage explicitly states the answer. Partial, implied
  or "probably" answers count as false.
- support: the numbers of every passage that states the answer on its own.
- answer: the answer in at most 30 words if answerable, else null.
- quote: if answerable, the shortest exact span (at most 40 words) of the first supporting
  passage that states the answer, copied character for character; else null.
- ambiguous: true if the question is unclear, or could refer to different things in different
  passages.

Reply with JSON only: {"answerable": true, "support": [1], "answer": "...", "quote": "...", "ambiguous": false}"""

V2_SYSTEM = """You check a training example for a model that must work only from the passages it is given.
Using ONLY the passage below:
- If the instruction is a yes/no question, answer it yourself: "Yes" or "No".
- Otherwise, decide whether every factual claim in the proposed response is stated in the passage
  ("supported"), and whether the response does what the instruction asks ("on_task").

Reply with JSON only: {"yes_no": "Yes" or "No" or null, "supported": true or false or null, "on_task": true or false or null}"""

J1_SYSTEM = """You grade a small model's answer. Use ONLY the passages; ignore outside knowledge.
- correct: the model's answer means the same as the reference answer. Missing the key fact, or
  adding a wrong claim, makes it false.
- supported: every factual claim in the model's answer is stated in the passages.

Reply with JSON only: {"correct": true or false, "supported": true or false}"""


def passages_block(passages: list[str]) -> str:
    return "\n".join(f"Passage {i}: {p}" for i, p in enumerate(passages, 1))


def gen_messages(kind: str, source: str, passage: str, task: str | None = None) -> tuple[str, str]:
    """(system, user) for G1 / G2 / G3."""
    desc = SOURCE_DESC[source]
    system = {"G1": G1_SYSTEM, "G2": G2_SYSTEM, "G3": G3_SYSTEM}[kind]
    system = system.format(source_desc=desc, task=task or "")
    return system, GEN_USER.format(source_desc=desc, passage=passage)


def v1_messages(passages: list[str], question: str) -> tuple[str, str]:
    body = passages_block(passages) if passages else "(no passages)"
    return V1_SYSTEM, f"{body}\nQuestion: {question}"


def v2_messages(k: int, passage: str, instruction: str, response: str | None) -> tuple[str, str]:
    user = f"Passage {k}: {passage}\nInstruction: {instruction}"
    if response is not None:
        user += f"\nProposed response: {response}"
    return V2_SYSTEM, user


def j1_messages(passages: list[str], question: str, reference: str, model_answer: str) -> tuple[str, str]:
    return J1_SYSTEM, (f"{passages_block(passages)}\nQuestion: {question}\n"
                       f"Reference answer: {reference}\nModel answer: {model_answer}")
