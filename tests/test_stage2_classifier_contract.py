"""K-MIND detached-z Stage-2 contract tests."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]


class Stage2ClassifierContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "kmind_teacher_audio_model",
            ROOT / "K-MIND/teacher_audio_model.py",
        )
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_projected_anchor_weight_is_bias_free_cosine_classifier(self):
        torch.manual_seed(3)
        projected_anchors = torch.randn(7, 12)
        args = SimpleNamespace(temp=0.15)
        classifier = self.module.Classifier(args, projected_anchors.clone())

        self.assertEqual(list(dict(classifier.named_parameters())), ["weight"])
        self.assertTrue(torch.equal(classifier.weight.detach(), projected_anchors))

        z = torch.randn(4, 12)
        expected = (
            (1 + F.cosine_similarity(
                projected_anchors.unsqueeze(0), z.unsqueeze(1), dim=-1
            )) / 2 + 1e-8
        ) / args.temp
        self.assertTrue(torch.allclose(classifier(z), expected))

    def test_refined_weight_is_used_for_final_inference(self):
        anchors = torch.eye(4)[:3]
        classifier = self.module.Classifier(SimpleNamespace(temp=0.5), anchors.clone())
        z = torch.eye(4)[:2]
        before = classifier(z).detach().clone()
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.1)
        optimizer.zero_grad()
        classifier(z).sum().backward()
        optimizer.step()
        after = classifier(z).detach()
        self.assertFalse(torch.equal(classifier.weight.detach(), anchors))
        self.assertFalse(torch.equal(before, after))

    def test_stage2_embeddings_are_built_without_gradients(self):
        source = (ROOT / "K-MIND/ELACL_fusion.py").read_text()
        self.assertIn("@torch.no_grad()\ndef build_stage2_embeddings_from_cache", source)
        self.assertIn("masked_mapped_output.detach().cpu()", source)


if __name__ == "__main__":
    unittest.main()
