"""IEMOCAP excluded-label dialogue-boundary regression test."""
import csv
import importlib.util
from pathlib import Path
import pickle
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "iemocap_preprocessing",
    ROOT / "IEMOCAP/preprocessing.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class IemocapPreprocessingBoundaryTest(unittest.TestCase):
    def test_minus_one_at_new_dialogue_does_not_leak_old_context(self):
        fields = [
            "Utterance", "Speaker", "Emotion", "Dialogue_ID", "Wav_Path",
            "Video_Path", "Start_Time", "End_Time",
        ]
        rows = [
            ["d1 valid", "A", "ang", "d1", "/x/u1.wav", "/x/d1.avi", "0", "1"],
            ["d1 skip", "B", "-1", "d1", "", "", "1", "2"],
            ["d2 skip", "C", "-1", "d2", "", "", "0", "1"],
            ["d2 valid", "D", "neu", "d2", "/x/u2.wav", "/x/d2.avi", "1", "2"],
        ]
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "split.csv"
            pickle_path = Path(directory) / "faces.pkl"
            with open(csv_path, "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(fields)
                writer.writerows(rows)
            with open(pickle_path, "wb") as handle:
                pickle.dump({"u1": {}, "u2": {}}, handle)
            samples = MODULE.preprocessing_pickle(str(csv_path), str(pickle_path))

        self.assertEqual(len(samples), 4)
        self.assertEqual([turn[1] for turn in samples[2]], ["d2 skip"])
        self.assertEqual([turn[1] for turn in samples[3]], ["d2 skip", "d2 valid"])
        self.assertNotIn("d1 valid", [turn[1] for turn in samples[3]])
        self.assertEqual(samples[2][-1][0], 0)
        self.assertEqual(samples[3][-1][0], 1)


if __name__ == "__main__":
    unittest.main()
