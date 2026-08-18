#!/usr/bin/env python3
"""Fail fast when a MELD CSV points to missing or empty frame directories."""
import argparse
import csv
import json
from pathlib import Path


def audit_csv(path):
    failures = []
    with open(path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            video_path = Path(row["Video_Path"])
            frame_dir = video_path.with_suffix("")
            frames = list(frame_dir.glob("*.jpg")) + list(frame_dir.glob("*.jpeg")) + list(frame_dir.glob("*.png"))
            if not frame_dir.is_dir():
                failures.append({"kind": "missing_frame_directory", "video_path": str(video_path)})
            elif not frames:
                failures.append({"kind": "empty_frame_directory", "video_path": str(video_path)})
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--dev", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    failures = {
        split: audit_csv(path)
        for split, path in (("train", args.train), ("dev", args.dev), ("test", args.test))
    }
    report = {"splits": failures, "failure_count": sum(map(len, failures.values()))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    if report["failure_count"]:
        raise SystemExit(f"HARD FAILURE: MELD media audit found {report['failure_count']} missing/empty frame directories")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
