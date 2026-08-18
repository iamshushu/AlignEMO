"""Regression tests for sample-independent pairwise HMF attention."""
import importlib.util
from pathlib import Path
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_module(relative_path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PairwiseAttentionTest(unittest.TestCase):
  def test_pairwise_attention_is_independent_of_batch_composition_and_order(self):
    torch.manual_seed(123)
    for relative_path, name in [
        ("IEMOCAP/model.py", "iemocap_model"),
        ("MELD/model.py", "meld_model"),
        ("K-MIND/model.py", "kmind_model"),
    ]:
        module = load_module(relative_path, name)
        attention = module.PairwiseModalityAttention(768, 4).eval()
        x = (torch.randn(768), torch.randn(768))
        y = (torch.randn(768), torch.randn(768))
        output_x_alone = attention(*(t.unsqueeze(0) for t in x))[0]
        output_xy = attention(*(torch.stack((a, b)) for a, b in zip(x, y)))[0]
        output_yx = attention(*(torch.stack((b, a)) for a, b in zip(x, y)))[1]
        self.assertTrue(torch.allclose(output_x_alone, output_xy, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(output_xy, output_yx, atol=1e-6, rtol=1e-6))


  def test_pairwise_attention_preserves_flattened_dimension(self):
      module = load_module("IEMOCAP/model.py", "iemocap_model_dimension")
      output = module.PairwiseModalityAttention(768, 4)(torch.randn(3, 768), torch.randn(3, 768))
      self.assertEqual(output.shape, (3, 1536))

  def test_kmind_active_entrypoint_is_batch_independent(self):
      module = load_module("K-MIND/teacher_audio_model.py", "kmind_active_model")
      attention = module.PairwiseModalityAttention(1280, 8).eval()
      x = (torch.randn(1280), torch.randn(1280))
      y = (torch.randn(1280), torch.randn(1280))
      alone = attention(*(item.unsqueeze(0) for item in x))[0]
      batched = attention(*(torch.stack((a, b)) for a, b in zip(x, y)))[0]
      reordered = attention(*(torch.stack((b, a)) for a, b in zip(x, y)))[1]
      self.assertTrue(torch.allclose(alone, batched, atol=1e-6, rtol=1e-6))
      self.assertTrue(torch.allclose(batched, reordered, atol=1e-6, rtol=1e-6))

  def test_ablation_variants_are_independent_of_batch_composition_and_order(self):
      module = load_module(
          "scripts/ablation/fusion_variants.py", "fusion_ablation_variants"
      )
      for variant in ("hmf_nogate", "shiftaware"):
          model = module.build_fusion(
              variant,
              clsNum=4,
              hidden_size=768,
              beta_shift=0.2,
              dropout_prob=0.0,
              num_head=4,
          ).eval()
          attention = model.pairwise_attn
          x = (torch.randn(768), torch.randn(768))
          y = (torch.randn(768), torch.randn(768))
          alone = attention(*(item.unsqueeze(0) for item in x))[0]
          batched = attention(
              *(torch.stack((a, b)) for a, b in zip(x, y))
          )[0]
          reordered = attention(
              *(torch.stack((b, a)) for a, b in zip(x, y))
          )[1]
          self.assertTrue(torch.allclose(alone, batched, atol=1e-6, rtol=1e-6))
          self.assertTrue(torch.allclose(batched, reordered, atol=1e-6, rtol=1e-6))

  def test_ablation_pairwise_attention_preserves_downstream_dimension(self):
      module = load_module(
          "scripts/ablation/fusion_variants.py", "fusion_ablation_dimensions"
      )
      output = module.PairwiseModalityAttention(768, 4)(
          torch.randn(3, 768), torch.randn(3, 768)
      )
      self.assertEqual(output.shape, (3, 1536))


if __name__ == "__main__":
    unittest.main()
