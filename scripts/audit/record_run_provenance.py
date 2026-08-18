#!/usr/bin/env python3
"""Write immutable, run-local provenance before a submitted experiment starts."""
import argparse
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def output(command):
    return subprocess.run(command, text=True, capture_output=True, check=False).stdout.strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--dataset", required=True)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--command", required=True)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--slurm-job-id", default=os.environ.get("SLURM_JOB_ID", ""))
    args = p.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "dataset": args.dataset, "seed": args.seed, "command": args.command,
        "slurm_job_id": args.slurm_job_id, "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": output(["git", "rev-parse", "HEAD"]),
        "source_diff_hash": hashlib.sha256(output(["git", "diff", "--binary"]).encode()).hexdigest(),
        "split_manifest": str(args.manifest) if args.manifest else None,
        "split_manifest_sha256": digest(args.manifest) if args.manifest else None,
        "python": platform.python_version(), "platform": platform.platform(),
    }
    target = args.run_dir / "provenance.json"
    with open(target, "x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(target)


if __name__ == "__main__":
    main()
