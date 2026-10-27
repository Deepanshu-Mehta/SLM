"""Phase 12: start jobs on the DEPLOYED rlaif app with .spawn(), so they run on Modal whatever the laptop does
(C-17: idle/lid sleep cancels `modal run --detach` jobs). Call ids are kept in .smoke/rlaif/calls.json.

    modal deploy rlaif_app.py                                    # once (private; no web endpoints; $0 idle)
    python rlaif_launch.py spawn sample_slm 0.18 --confirm       # function, --max-usd (stage caps in config)
    python rlaif_launch.py status                                # done / running / failed per call
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import modal

import config

APP = f"{config.PROJECT}-rlaif"
CALLS = Path(".smoke/rlaif/calls.json")
STAGE = {"sample_slm": "12.3-sample-gpu", "sample_g3": "12.3-sample-gpu", "headroom_slm": "12.1-headroom-gpu",
         "headroom_g3": "12.1-headroom-gpu", "dpo_slm": "12.5-dpo-gpu", "dpo_g3": "12.5-dpo-gpu"}


def load() -> list[dict]:
    return json.loads(CALLS.read_text()) if CALLS.exists() else []


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("spawn", "status"))
    p.add_argument("fn", nargs="?")
    p.add_argument("max_usd", nargs="?", type=float)
    p.add_argument("--confirm", action="store_true")
    a = p.parse_args()
    calls = load()
    if a.cmd == "spawn":
        if not a.confirm:
            raise SystemExit("spawning a GPU job costs money: re-run with --confirm once approved")
        cap = config.RLAIF_STAGE_CAPS_USD[STAGE[a.fn]]
        if a.max_usd > cap:
            raise SystemExit(f"--max-usd {a.max_usd} exceeds the stage cap ${cap}")
        c = modal.Function.from_name(APP, a.fn).spawn(a.max_usd)
        calls.append({"fn": a.fn, "max_usd": a.max_usd, "id": c.object_id, "t": time.time()})
        CALLS.parent.mkdir(parents=True, exist_ok=True)
        CALLS.write_text(json.dumps(calls, indent=2))
        print(f"spawned {a.fn} (max ${a.max_usd}) -> {c.object_id}")
        return
    for c in calls:
        fc = modal.FunctionCall.from_id(c["id"])
        try:
            res = fc.get(timeout=0)
            state = "done: " + json.dumps({k: v for k, v in res.items() if k not in ("runs", "gens")})[:300]
        except TimeoutError:
            state = "running"
        except Exception as e:                       # noqa: BLE001 (report any remote failure)
            state = f"failed: {type(e).__name__}: {str(e)[:200]}"
        print(f"{c['fn']:14s} {c['id']}  {state}")


if __name__ == "__main__":
    main()
