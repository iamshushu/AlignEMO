"""Regression tests for immutable split-manifest overlap handling."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit/validate_splits.py"


class ValidateSplitsTest(unittest.TestCase):
    def write_csv(self, directory, name, rows):
        path = directory / name
        path.write_text("Dialogue_ID,Utterance_ID,Speaker,Emotion,Wav_Path\n" + "\n".join(rows) + "\n")
        return path

    def write_meld_csv(self, directory, name, video_path):
        path = directory / name
        path.write_text(
            "Dialogue_ID,Utterance_ID,Season,Episode,Speaker,Emotion,Video_Path\n"
            f"58,0,4,13,speaker,happy,{video_path}\n"
        )
        return path

    def test_dialogue_overlap_is_hard_failure_and_speaker_overlap_is_warning(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            train = self.write_csv(directory, "train.csv", ["d1,u1,same,happy,a.wav"])
            dev = self.write_csv(directory, "dev.csv", ["d2,u1,same,sad,b.wav"])
            test = self.write_csv(directory, "test.csv", ["d1,u2,other,angry,c.wav"])
            manifest = directory / "manifest.json"
            result = subprocess.run([sys.executable, str(SCRIPT), "--dataset", "fixture", "--train", str(train),
                                     "--dev", str(dev), "--test", str(test), "--output", str(manifest)],
                                    text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            payload = json.loads(manifest.read_text())
            self.assertEqual(payload["hard_failures"][0]["kind"], "dialogues")
            self.assertEqual(payload["speaker_overlap_warnings"][0]["count"], 1)

    def test_nonoverlapping_manifest_is_created_once(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            paths = [self.write_csv(directory, f"{name}.csv", [f"d{i},u1,s{i},happy,{name}.wav"])
                     for i, name in enumerate(("train", "dev", "test"))]
            manifest = directory / "manifest.json"
            command = [sys.executable, str(SCRIPT), "--dataset", "fixture", "--train", str(paths[0]),
                       "--dev", str(paths[1]), "--test", str(paths[2]), "--output", str(manifest),
                       "--expected-rows", "train=1"]
            self.assertEqual(subprocess.run(command, text=True, capture_output=True).returncode, 0)
            self.assertNotEqual(subprocess.run(command, text=True, capture_output=True).returncode, 0)

    def test_meld_split_local_dialogue_ids_use_video_path_identity(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            paths = [
                self.write_meld_csv(directory, "train.csv", "train_splits/dia58_utt0.mp4"),
                self.write_meld_csv(directory, "dev.csv", "dev_splits/dia58_utt0.mp4"),
                self.write_meld_csv(directory, "test.csv", "test_splits/dia58_utt0.mp4"),
            ]
            manifest = directory / "manifest.json"
            result = subprocess.run([sys.executable, str(SCRIPT), "--dataset", "MELD", "--train", str(paths[0]),
                                     "--dev", str(paths[1]), "--test", str(paths[2]), "--output", str(manifest)],
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
