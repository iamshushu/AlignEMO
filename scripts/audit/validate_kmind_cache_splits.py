#!/usr/bin/env python3
"""Build an immutable K-MIND split manifest from rowwise cache metadata.

Dialogue/utterance overlap is a hard failure.  Speaker overlap is reported as
a warning because the same speaker labels are expected across conversations.
The script never changes cache files and refuses to overwrite its output.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def stable_media_identity(meta):
    raw = str(meta.get("video_path") or meta.get("wav_path") or "").strip()
    path = Path(raw)
    if not raw:
        return None, None
    if path.name == "audio.wav":
        utterance = path.parent.name
        dialogue = path.parent.parent.name
    else:
        utterance = path.name
        dialogue = path.parent.name
    if not dialogue or not utterance:
        return None, None
    return dialogue, f"{dialogue}/{utterance}"


def read_cache_split(cache_root, split, torch):
    split_dir = cache_root / split
    files = sorted(split_dir.glob("row_*.pt"))
    if not files:
        raise ValueError(f"empty or missing cache split: {split_dir}")
    dialogues, utterances, speakers = set(), set(), set()
    labels, missing_identity = Counter(), []
    fingerprint = hashlib.sha256()
    for path in files:
        row = torch.load(path, map_location="cpu")
        meta = row.get("meta", {})
        dialogue, utterance = stable_media_identity(meta)
        if dialogue is None:
            missing_identity.append(path.name)
        else:
            dialogues.add(dialogue)
            utterances.add(utterance)
        speaker = str(meta.get("speaker", "")).strip()
        if speaker:
            speakers.add(speaker)
        label = int(row["y"].item())
        labels[label] += 1
        identity_record = {
            "file": path.name,
            "dialogue": dialogue,
            "utterance": utterance,
            "speaker": speaker,
            "label": label,
        }
        fingerprint.update(
            (json.dumps(identity_record, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        )
    return {
        "rows": len(files),
        "dialogues": sorted(dialogues),
        "utterances": sorted(utterances),
        "speakers": sorted(speakers),
        "class_distribution": {str(key): labels[key] for key in sorted(labels)},
        "missing_identity_count": len(missing_identity),
        "missing_identity_examples": missing_identity[:20],
        "identity_manifest_sha256": fingerprint.hexdigest(),
    }


def overlap(left, right):
    return sorted(set(left) & set(right))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-rows", action="append", default=[], metavar="SPLIT=COUNT")
    args = parser.parse_args()
    import torch

    splits = {
        split: read_cache_split(args.cache_root, split, torch)
        for split in ("train", "dev", "test")
    }
    failures, speaker_warnings = [], []
    for split, record in splits.items():
        if record["missing_identity_count"]:
            failures.append({
                "kind": "missing_identity",
                "split": split,
                "count": record["missing_identity_count"],
                "examples": record["missing_identity_examples"],
            })
    for item in args.expected_rows:
        try:
            split, expected_text = item.split("=", 1)
            expected = int(expected_text)
            actual = splits[split]["rows"]
        except (ValueError, KeyError) as error:
            raise SystemExit(f"invalid --expected-rows {item!r}; expected SPLIT=COUNT") from error
        if actual != expected:
            failures.append({
                "kind": "row_count", "split": split,
                "expected": expected, "actual": actual,
            })
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        for key in ("dialogues", "utterances"):
            shared = overlap(splits[left][key], splits[right][key])
            if shared:
                failures.append({
                    "kind": key, "splits": [left, right],
                    "count": len(shared), "examples": shared[:20],
                })
        shared_speakers = overlap(splits[left]["speakers"], splits[right]["speakers"])
        if shared_speakers:
            speaker_warnings.append({
                "splits": [left, right], "count": len(shared_speakers),
                "examples": shared_speakers[:20],
            })
    manifest = {
        "dataset": "K-MIND",
        "source": str(args.cache_root.resolve()),
        "identity_rule": "dialogue=media parent directory; utterance=dialogue/media leaf",
        "splits": splits,
        "hard_failures": failures,
        "speaker_overlap_warnings": speaker_warnings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({
        "manifest": str(args.output),
        "rows": {key: value["rows"] for key, value in splits.items()},
        "hard_failure_count": len(failures),
        "speaker_overlap_warnings": speaker_warnings,
    }, ensure_ascii=False))
    if failures:
        raise SystemExit(f"HARD FAILURE: K-MIND split validation failed; see {args.output}")


if __name__ == "__main__":
    main()
