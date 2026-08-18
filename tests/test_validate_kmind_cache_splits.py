"""Unit tests for K-MIND cache identity extraction and overlap policy."""
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "validate_kmind_cache_splits",
    ROOT / "scripts/audit/validate_kmind_cache_splits.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ValidateKmindCacheSplitsTest(unittest.TestCase):
    def test_video_path_identity(self):
        meta = {"video_path": "/root/frames/SET037_M005_S002/00102"}
        self.assertEqual(
            MODULE.stable_media_identity(meta),
            ("SET037_M005_S002", "SET037_M005_S002/00102"),
        )

    def test_audio_path_fallback_identity(self):
        meta = {"wav_path": "/root/wavs/SET062_M006_S003/00164/audio.wav"}
        self.assertEqual(
            MODULE.stable_media_identity(meta),
            ("SET062_M006_S003", "SET062_M006_S003/00164"),
        )

    def test_overlap_is_exact_set_intersection(self):
        self.assertEqual(MODULE.overlap(["d2", "d1"], ["d3", "d2"]), ["d2"])


if __name__ == "__main__":
    unittest.main()
