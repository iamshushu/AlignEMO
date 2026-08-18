"""K-MIND active entrypoint must select an explicit 768/1280 contract."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_entrypoint():
    spec = importlib.util.spec_from_file_location(
        "kmind_elacl_contract", ROOT / "K-MIND" / "ELACL_fusion.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class KmindModelContractSelectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_entrypoint()

    def test_paper_contract_selects_768_models(self):
        visual, fusion, cl_model, classifier = self.module.resolve_model_classes(
            SimpleNamespace(representation_dim=768, final_fusion_variant="hmf")
        )
        self.assertEqual(visual.__module__, "kmind_paper_models_768")
        self.assertEqual(fusion.__module__, "kmind_paper_models_768")
        self.assertEqual(cl_model.__module__, "kmind_paper_models_768")
        self.assertEqual(classifier.__module__, "kmind_paper_models_768")

    def test_nogate_is_available_only_on_corrected_contract(self):
        _, fusion, _, _ = self.module.resolve_model_classes(
            SimpleNamespace(representation_dim=768, final_fusion_variant="hmf_nogate")
        )
        self.assertEqual(fusion.__name__, "HMFNoGate")
        with self.assertRaises(ValueError):
            self.module.resolve_model_classes(
                SimpleNamespace(representation_dim=1280, final_fusion_variant="hmf_nogate")
            )

    def test_legacy_contract_remains_explicitly_available(self):
        visual, fusion, cl_model, classifier = self.module.resolve_model_classes(
            SimpleNamespace(representation_dim=1280, final_fusion_variant="hmf")
        )
        for item in (visual, fusion, cl_model, classifier):
            self.assertEqual(item.__module__, "kmind_legacy_models_1280")

    def test_scheduler_counts_optimizer_not_microbatch_steps(self):
        self.assertEqual(self.module.optimizer_training_steps(100, 15, 8), 195)
        self.assertEqual(self.module.optimizer_training_steps(100, 15, 2), 750)
        self.assertEqual(self.module.optimizer_training_steps(100, 15, 1), 1500)
        with self.assertRaises(ValueError):
            self.module.optimizer_training_steps(100, 15, 0)


if __name__ == "__main__":
    unittest.main()
