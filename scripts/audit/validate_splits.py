#!/usr/bin/env python3
"""Validate immutable train/dev/test CSV split manifests without changing data."""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_present(fieldnames, candidates):
    lowered = {name.lower().lstrip("\ufeff"): name for name in fieldnames}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def dialogue_identity(row, dialogue, season, episode, video):
    """Return a split-stable dialogue key without conflating MELD local IDs.

    MELD restarts ``Dialogue_ID`` in each official split.  When an explicit
    video path is available, its split directory plus the ``diaN`` stem is the
    authoritative dialogue identity; Season/Episode/Dialogue_ID remains the
    fallback for datasets without media paths.
    """
    if video and row.get(video, "").strip():
        video_path = Path(row[video].strip())
        dialogue_stem = re.sub(r"_utt\\d+$", "", video_path.stem)
        return str(video_path.parent / dialogue_stem)
    return "::".join(part for part in (
        row.get(season, "").strip() if season else "",
        row.get(episode, "").strip() if episode else "",
        row[dialogue].strip(),
    ) if part)


def read_split(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty split: {path}")
    fields = rows[0].keys()
    dialogue = first_present(fields, ["Dialogue_ID", "dialogue_id"])
    season = first_present(fields, ["Season", "season"])
    episode = first_present(fields, ["Episode", "episode"])
    speaker = first_present(fields, ["Speaker", "speaker"])
    utterance = first_present(fields, ["Utterance_ID", "utterance_id", "Utterance"])
    label = first_present(fields, ["Emotion", "Affective_State", "label", "emotion"])
    video = first_present(fields, ["Video_Path", "video_path"])
    media = [first_present(fields, ["Wav_Path", "wav_path"]), video]
    if dialogue is None or utterance is None:
        raise ValueError(f"{path}: requires dialogue and utterance columns; found {list(fields)}")
    values = lambda key: {row[key].strip() for row in rows if key and row.get(key, "").strip()}
    dialogue_ids = {
        dialogue_identity(row, dialogue, season, episode, video)
        for row in rows if row.get(dialogue, "").strip()
    }
    utterance_ids = {
        f"{dialogue_identity(row, dialogue, season, episode, video)}::{row[utterance].strip()}"
        for row in rows
        if row.get(dialogue, "").strip() and row.get(utterance, "").strip()
    }
    return {
        "csv_sha256": sha256(path), "rows": len(rows), "dialogues": sorted(dialogue_ids),
        "utterances": sorted(utterance_ids), "speakers": sorted(values(speaker)),
        "media": sorted(set().union(*(values(key) for key in media if key))),
        "class_distribution": dict(sorted(Counter(row[label] for row in rows if label).items())),
    }


def overlap(left, right):
    return sorted(set(left) & set(right))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--dev", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-rows", action="append", default=[], metavar="SPLIT=COUNT",
                        help="Optional Table-1/loader count assertion, e.g. --expected-rows train=100")
    args = parser.parse_args()
    splits = {name: read_split(path) for name, path in (("train", args.train), ("dev", args.dev), ("test", args.test))}
    failures, speaker_warnings = [], []
    for item in args.expected_rows:
        try:
            split, expected = item.split("=", 1)
            expected = int(expected)
            actual = splits[split]["rows"]
        except (ValueError, KeyError) as error:
            raise SystemExit(f"invalid --expected-rows {item!r}; expected SPLIT=COUNT") from error
        if actual != expected:
            failures.append({"kind": "row_count", "split": split, "expected": expected, "actual": actual})
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        for key in ("dialogues", "utterances", "media"):
            shared = overlap(splits[left][key], splits[right][key])
            if shared:
                failures.append({"kind": key, "splits": [left, right], "examples": shared[:20], "count": len(shared)})
        shared_speakers = overlap(splits[left]["speakers"], splits[right]["speakers"])
        if shared_speakers:
            speaker_warnings.append({"splits": [left, right], "examples": shared_speakers[:20], "count": len(shared_speakers)})
    manifest = {"dataset": args.dataset, "splits": splits, "hard_failures": failures, "speaker_overlap_warnings": speaker_warnings}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    if failures:
        raise SystemExit(f"HARD FAILURE: split overlap found; manifest written to {args.output}")
    print(json.dumps({"manifest": str(args.output), "speaker_overlap_warnings": speaker_warnings}, ensure_ascii=False))


if __name__ == "__main__":
    main()
