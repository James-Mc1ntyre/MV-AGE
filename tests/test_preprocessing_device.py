import builtins
import sys
import unittest
from unittest import mock

import numpy as np

from mv_age.Preprocessing import (
    build_view_graph,
    encode_ingredient_lists,
    resolve_sentence_encoder_device,
)


class TestSentenceEncoderDeviceResolution(unittest.TestCase):
    def test_auto_uses_cuda_when_torch_reports_gpu(self):
        fake_torch = mock.Mock()
        fake_torch.cuda.is_available.return_value = True

        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            self.assertEqual(resolve_sentence_encoder_device("auto"), "cuda")

    def test_auto_uses_cpu_when_torch_reports_no_gpu(self):
        fake_torch = mock.Mock()
        fake_torch.cuda.is_available.return_value = False

        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            self.assertEqual(resolve_sentence_encoder_device("auto"), "cpu")

    def test_explicit_cuda_raises_when_torch_reports_no_gpu(self):
        fake_torch = mock.Mock()
        fake_torch.cuda.is_available.return_value = False

        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            with self.assertRaisesRegex(RuntimeError, "torch.cuda.is_available\\(\\) is False"):
                resolve_sentence_encoder_device("cuda")

    def test_sentence_transformers_import_error_includes_original_exception(self):
        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "sentence_transformers":
                raise ImportError("broken torch dependency")
            return real_import(name, globals, locals, fromlist, level)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaisesRegex(ImportError, "Original import error: broken torch dependency"):
                encode_ingredient_lists(["salt, water"], model_name="all-MiniLM-L6-v2")

    def test_build_view_graph_clamps_neighbors_for_small_dataset(self):
        x = np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=np.float32,
        )

        graph = build_view_graph(x, n_neighbors=30)

        self.assertEqual(graph.shape, (3, 3))
        self.assertGreater(graph.nnz, 0)


if __name__ == "__main__":
    unittest.main()
