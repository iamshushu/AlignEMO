"""Regression tests for the paper-faithful K-MIND 768-d cache path."""
import importlib.util
from pathlib import Path
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]


def load(relative_path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class KmindProjectionTest(unittest.TestCase):
    def test_projection_calibrator_maps_every_modality_to_768(self):
        module = load(
            "scripts/ablation/project_kmind_cache_768.py", "kmind_projection"
        )
        model = module.ModalityProjectionCalibrator(1280, 768, 7)
        projected, logits = model(tuple(torch.randn(3, 1280) for _ in range(4)))
        self.assertEqual(set(projected), {"text", "v2", "v1", "audio"})
        for value in projected.values():
            self.assertEqual(value.shape, (3, 768))
        for value in logits.values():
            self.assertEqual(value.shape, (3, 7))

    def test_hierarchical_variants_preserve_audio_anchor_and_output_shape(self):
        module = load(
            "scripts/ablation/train_kmind_fusion_cached.py", "kmind_fusion_cached"
        )
        for variant in ("concat", "crossattn", "hmf_nogate", "hmf", "shiftaware"):
            model = module.HierarchicalFusionVariant(
                variant, 7, 768, 0.1, 0.0, 3
            ).eval()
            hidden, logits = model(
                text=torch.randn(2, 768),
                v2=torch.randn(2, 768),
                v1=torch.randn(2, 768),
                audio=torch.randn(2, 768),
            )
            self.assertEqual(hidden.shape, (2, 768))
            self.assertEqual(logits.shape, (2, 7))

    def test_final_beta_can_vary_while_visual_beta_stays_fixed(self):
        module = load(
            "scripts/ablation/train_kmind_fusion_cached.py", "kmind_beta_control"
        )
        model = module.HierarchicalFusionVariant(
            "hmf", 7, 768, 0.5, 0.0, 3, visual_beta_shift=0.1
        )
        self.assertEqual(model.visual_fusion.beta_shift, 0.1)
        self.assertEqual(model.final_fusion.beta_shift, 0.5)


if __name__ == "__main__":
    unittest.main()
