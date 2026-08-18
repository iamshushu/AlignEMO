"""Regression test: training loops must not evaluate the test split per epoch."""
import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EXPLICIT_TRAINERS = (
    "IEMOCAP/teacher.py",
    "IEMOCAP/student_audio.py",
    "IEMOCAP/student_visual.py",
    "IEMOCAP/student_text.py",
    "IEMOCAP/audio_teacher.py",
    "MELD/teacher.py",
    "MELD/student_audio.py",
    "MELD/student_visual.py",
    "MELD/student_text.py",
    "MELD/audio_teacher.py",
    "K-MIND/teacher_audio_ekman.py",
    "K-MIND/teacher_ekman.py",
    "K-MIND/student_visual_ekman.py",
)


def is_test_evaluation(node):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "evaluation"
        and any(
            isinstance(argument, ast.Name)
            and argument.id in {"test_dataloader", "test_loader"}
            for argument in node.args
        )
    )


def is_epoch_loop(node):
    if not isinstance(node, (ast.For, ast.While)):
        return False
    names = {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}
    return "epoch" in names or "training_epochs" in names


class ValidationOnlySelectionTest(unittest.TestCase):
    def test_test_split_is_evaluated_once_after_the_epoch_loop(self):
        for relative_path in EXPLICIT_TRAINERS:
            tree = ast.parse((ROOT / relative_path).read_text())
            test_evaluations = [
                node for node in ast.walk(tree) if is_test_evaluation(node)
            ]
            self.assertEqual(len(test_evaluations), 1, relative_path)

            for epoch_loop in (node for node in ast.walk(tree) if is_epoch_loop(node)):
                self.assertFalse(
                    any(is_test_evaluation(node) for node in ast.walk(epoch_loop)),
                    relative_path,
                )

    def test_no_test_loader_call_exists_inside_any_epoch_loop(self):
        for dataset in ("IEMOCAP", "MELD", "K-MIND"):
            for path in (ROOT / dataset).glob("*.py"):
                tree = ast.parse(path.read_text())
                for epoch_loop in (node for node in ast.walk(tree) if is_epoch_loop(node)):
                    for call in (node for node in ast.walk(epoch_loop) if isinstance(node, ast.Call)):
                        names = {
                            child.id for child in ast.walk(call)
                            if isinstance(child, ast.Name)
                        }
                        self.assertTrue(
                            names.isdisjoint({"test_dataloader", "test_loader"}),
                            f"{path.relative_to(ROOT)}:{call.lineno}",
                        )

    def test_kmind_stage2_sweep_can_avoid_loading_test(self):
        source = (ROOT / "K-MIND" / "ELACL_fusion.py").read_text()
        self.assertIn("--stage2_validation_only", source)
        self.assertIn("test samples:  NOT LOADED (--stage2_validation_only)", source)
        self.assertIn('"test_loaded_or_evaluated": False', source)


if __name__ == "__main__":
    unittest.main()
