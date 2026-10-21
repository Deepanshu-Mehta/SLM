"""Copy the Modal Volume to a new Modal account, volume to volume inside Modal (2026-10-21).

    bash -c 'set -a; source .env.local; set +a; .venv/bin/modal run migrate_app.py'

Runs in the NEW account (MODAL_TOKEN_ID/SECRET). Each container reads one file from the OLD
account's Volume with the old token (MODAL_OLD_TOKEN_ID/SECRET, passed as an ephemeral secret,
never stored) and writes it to the new Volume of the same name. One worker per file; a file
already present with the same size is skipped, so a rerun resumes. `clean/` (superseded by
`corpus/`) and `checkpoints/calibration/` (throwaway calibration checkpoints) are not copied.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

import config

SKIP = ("clean/", "checkpoints/calibration/")

app = modal.App(f"{config.PROJECT}-migrate")
image = (modal.Image.debian_slim(python_version="3.12").pip_install(f"modal=={modal.__version__}")
         .add_local_python_source("config"))
volume = modal.Volume.from_name(config.VOLUME_NAME, create_if_missing=True)
old_token = modal.Secret.from_dict({"OLD_ID": os.environ.get("MODAL_OLD_TOKEN_ID", ""),
                                    "OLD_SECRET": os.environ.get("MODAL_OLD_TOKEN_SECRET", "")})


def _old_volume():
    client = modal.Client.from_credentials(os.environ.get("OLD_ID") or os.environ["MODAL_OLD_TOKEN_ID"],
                                           os.environ.get("OLD_SECRET") or os.environ["MODAL_OLD_TOKEN_SECRET"])
    return modal.Volume.from_name(config.VOLUME_NAME, client=client)


@app.function(image=image, volumes={config.DATA_ROOT: volume}, secrets=[old_token], timeout=60 * 30,
              cpu=1.0, memory=2048, max_containers=24)
def copy_file(path: str, size: int) -> dict:
    dst = Path(config.DATA_ROOT) / path
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    with tmp.open("wb") as fh:
        _old_volume().read_file_into_fileobj(path, fh)
    got = tmp.stat().st_size
    if got != size:
        tmp.unlink()
        return {"path": path, "ok": False, "size": got, "expected": size}
    tmp.rename(dst)
    volume.commit()
    return {"path": path, "ok": True, "size": got}


@app.local_entrypoint()
def main():
    old = _old_volume()
    files = {e.path: e.size for e in old.listdir("/", recursive=True)
             if e.type == modal.volume.FileEntryType.FILE and not e.path.startswith(SKIP)}
    have = {e.path: e.size for e in volume.listdir("/", recursive=True)
            if e.type == modal.volume.FileEntryType.FILE}
    todo = sorted((p for p, s in files.items() if have.get(p) != s), key=lambda p: -files[p])
    print(f"{len(files)} files ({sum(files.values()) / 2**30:.2f} GiB) to mirror; "
          f"{len(files) - len(todo)} already present; copying {len(todo)}")
    bad = []
    for r in copy_file.starmap([(p, files[p]) for p in todo]):
        if not r["ok"]:
            bad.append(r)
        print(("ok  " if r["ok"] else "BAD ") + f"{r['path']} {r['size'] / 2**20:.1f} MiB", flush=True)
    have = {e.path: e.size for e in volume.listdir("/", recursive=True)
            if e.type == modal.volume.FileEntryType.FILE}
    missing = [p for p, s in files.items() if have.get(p) != s]
    print(f"verify: {len(files) - len(missing)}/{len(files)} files match by size; "
          f"{len(bad)} bad copies; missing {missing[:5]}")
