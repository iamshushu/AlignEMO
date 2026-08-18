"""Regression test for MELD missing-frame preflight checks."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit/validate_meld_media.py"


class ValidateMeldMediaTest(unittest.TestCase):
    def write_csv(self, directory, name, video_path):
        path = directory / name
        path.write_text("Video_Path\n" + video_path + "\n")
        return path

    def test_missing_and_empty_directories_are_hard_failures(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            (directory / "empty").mkdir()
            train = self.write_csv(directory, "train.csv", str(directory / "empty.mp4"))
            dev = self.write_csv(directory, "dev.csv", str(directory / "missing.mp4"))
            good = directory / "good"
            good.mkdir()
            (good / "frame_00000.jpg").write_bytes(b"fixture")
            test = self.write_csv(directory, "test.csv", str(directory / "good.mp4"))
            output = directory / "audit.json"
            result = subprocess.run([sys.executable, str(SCRIPT), "--train", str(train), "--dev", str(dev),
                                     "--test", str(test), "--output", str(output)], text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(json.loads(output.read_text())["failure_count"], 2)


if __name__ == "__main__":
    unittest.main()
