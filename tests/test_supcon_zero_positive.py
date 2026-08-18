"""K-MIND ELACL query-mask regression test for the documented B=4 case."""
import importlib.util
from pathlib import Path
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]


def reference_reductions(reps, labels, anchors, temperature, eps=1e-8):
  """Reproduce the current contrastive term and two candidate query reductions."""
  all_reps = torch.cat((reps, anchors), dim=0)
  all_labels = torch.cat((labels, torch.arange(len(anchors))))
  size = len(all_labels)
  nonself = 1 - torch.eye(size)
  positive_mask = (all_labels[:, None] == all_labels[None, :]).float()
  scores = (1 + torch.nn.functional.cosine_similarity(
      all_reps[:, None, :], all_reps[None, :, :], dim=-1
  )) / 2 + eps
  scores *= nonself
  scores /= temperature
  scores -= scores.max().item()
  scores = torch.exp(scores)
  positive_scores = scores * positive_mask * nonself
  negative_scores = scores * (1 - positive_mask)
  positive_counts = (positive_mask * nonself).sum(dim=1)
  probabilities = positive_scores.sum(-1) / (
      positive_scores.sum(-1) + negative_scores.sum(-1)
  )
  probabilities /= positive_counts + eps
  per_query = -torch.log(probabilities + eps)
  current_mask = per_query > 0
  valid_mask = positive_counts > 0
  batch_size = len(labels)
  return {
      "positive_counts": positive_counts,
      "per_query": per_query,
      "current": per_query[current_mask].mean(),
      "utterance_only": per_query[:batch_size].mean(),
      "valid_query": per_query[valid_mask].mean(),
  }


class SupConZeroPositiveTest(unittest.TestCase):
  def test_anchor_queries_can_be_zero_positive_but_loss_is_finite(self):
    torch.manual_seed(7)
    anchors = torch.randn(7, 768)
    original_load = torch.load
    torch.load = lambda *args, **kwargs: anchors.clone()
    try:
      spec = importlib.util.spec_from_file_location("kmind_loss", ROOT / "K-MIND/ELACL_loss.py")
      loss_module = importlib.util.module_from_spec(spec)
      spec.loader.exec_module(loss_module)

      class Args:
          temp = 0.5
          angle_loss_weight = 0.0
          disable_emo_anchor = False
          dataset_name = "K-MIND"
          anchor_path = "unused"
          supcon_query_mode = "current"

      class Model:
          def map_function_anchor(self, anchors):
              return anchors

      labels = torch.tensor([0, 1, 2, 3])
      features = torch.randn(4, 768, requires_grad=True)
      reductions = reference_reductions(features, labels, anchors, Args.temp)
      positive_counts = reductions["positive_counts"]
      self.assertEqual(positive_counts[:4].tolist(), [1, 1, 1, 1])
      self.assertEqual(positive_counts[4:].tolist(), [1, 1, 1, 1, 0, 0, 0])
      valid_query_mask = positive_counts > 0
      self.assertEqual(valid_query_mask.nonzero().flatten().tolist(), list(range(8)))
      self.assertEqual(valid_query_mask[:4].nonzero().flatten().tolist(), [0, 1, 2, 3])
      self.assertEqual((~valid_query_mask).nonzero().flatten().tolist(), [8, 9, 10])
      # The current `loss > 0` mask is not a positive-count mask: zero-positive
      # anchors receive approximately -log(eps) and enter the current reduction.
      self.assertTrue(torch.all(reductions["per_query"][8:] > 18.0))
      print("utterance positive counts:", positive_counts[:4].tolist())
      print("anchor positive counts:", positive_counts[4:].tolist())
      print("zero-positive query indices:", (~valid_query_mask).nonzero().flatten().tolist())
      print("current contrastive reduction:", float(reductions["current"].detach()))
      print("utterance-only reduction:", float(reductions["utterance_only"].detach()))
      print("valid-query reduction:", float(reductions["valid_query"].detach()))
      loss = loss_module.SupConLoss(Args())(features, labels, Model()).loss
      print("current full finite loss:", float(loss.detach()))
      self.assertTrue(torch.isfinite(loss))
      self.assertTrue(torch.isfinite(reductions["utterance_only"]))
      self.assertTrue(torch.isfinite(reductions["valid_query"]))
      self.assertGreater(
          float(reductions["current"].detach()),
          float(reductions["valid_query"].detach()),
      )
      loss.backward()

      for mode, expected_key in (
          ("current", "current"),
          ("valid_query", "valid_query"),
          ("utterance_only", "utterance_only"),
      ):
        Args.supcon_query_mode = mode
        actual = loss_module.SupConLoss(Args())(
            features.detach(), labels, Model()
        ).loss
        self.assertTrue(torch.allclose(actual, reductions[expected_key], atol=1e-6))
    finally:
      torch.load = original_load


if __name__ == "__main__":
    unittest.main()
