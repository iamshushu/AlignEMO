"""Regression test: IEMOCAP/MELD fusion calls must be fully keyword-based."""
import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "fusion": {"text_embedding", "visual", "acoustic"},
    "fusion_visual": {"text_embedding", "visual", "visual_2"},
}


class FusionKeywordArgumentsTest(unittest.TestCase):
    def test_all_fusion_calls_use_complete_keyword_arguments(self):
        found = {name: 0 for name in EXPECTED}
        paths = []
        for dataset in ("IEMOCAP", "MELD"):
            paths.extend((ROOT / dataset).glob("*.py"))
        paths.extend([
            ROOT / "scripts/ablation/build_fusion_cache.py",
            ROOT / "scripts/ablation/build_meld_fusion_cache.py",
            ROOT / "scripts/ablation/check_fusion_order.py",
            ROOT / "scripts/ablation/train_fusion_cached.py",
            ROOT / "scripts/ablation/train_fusion_variant.py",
            ROOT / "scripts/analysis/dump_predictions_iemocap.py",
            ROOT / "scripts/analysis/dump_predictions_meld.py",
        ])
        for path in paths:
            tree = ast.parse(path.read_text())
            for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
                if isinstance(call.func, ast.Name):
                    name = call.func.id
                elif isinstance(call.func, ast.Attribute):
                    name = call.func.attr
                else:
                    continue
                if name not in EXPECTED:
                    continue
                found[name] += 1
                self.assertEqual(call.args, [], f"{path.relative_to(ROOT)}:{call.lineno}")
                keywords = {keyword.arg for keyword in call.keywords}
                self.assertEqual(
                    keywords,
                    EXPECTED[name],
                    f"{path.relative_to(ROOT)}:{call.lineno}",
                )
        self.assertGreater(found["fusion"], 0)
        self.assertGreater(found["fusion_visual"], 0)


if __name__ == "__main__":
    unittest.main()
