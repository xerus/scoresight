"""Check the experimental trainer against enumeration and numerical gradients."""

import itertools
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/ocr_benchmark"))
from ctc_head import ctc_loss_gradient, log_softmax  # noqa: E402


class CtcHeadTests(unittest.TestCase):
    def test_loss_matches_all_alignments_including_repeated_and_blank_targets(self):
        logits = np.array([[0.2, 0.6, -0.4], [0.8, -0.1, 0.3], [-0.2, 0.5, 0.1]])
        probabilities = np.exp(log_softmax(logits))
        for target in ([], [1], [1, 1], [1, 2], [2, 1]):
            with self.subTest(target=target):
                total = 0.0
                for alignment in itertools.product(range(3), repeat=3):
                    collapsed = [
                        value
                        for t, value in enumerate(alignment)
                        if value and (t == 0 or value != alignment[t - 1])
                    ]
                    if collapsed == target:
                        total += np.prod(
                            [probabilities[t, c] for t, c in enumerate(alignment)]
                        )
                loss, gradient = ctc_loss_gradient(logits, target)
                self.assertAlmostEqual(loss, -np.log(total), places=10)
                np.testing.assert_allclose(gradient.sum(axis=1), 0, atol=1e-12)

    def test_gradient_matches_finite_differences(self):
        rng = np.random.default_rng(12)
        logits = rng.normal(size=(5, 3))
        epsilon = 1e-5
        for target in ([], [1], [1, 1], [1, 2]):
            with self.subTest(target=target):
                _, analytic = ctc_loss_gradient(logits, target)
                numeric = np.zeros_like(logits)
                for index in np.ndindex(logits.shape):
                    plus, minus = logits.copy(), logits.copy()
                    plus[index] += epsilon
                    minus[index] -= epsilon
                    numeric[index] = (
                        ctc_loss_gradient(plus, target)[0]
                        - ctc_loss_gradient(minus, target)[0]
                    ) / (2 * epsilon)
                np.testing.assert_allclose(analytic, numeric, atol=1e-8)

    def test_impossible_alignment_and_blank_target_class_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot be aligned"):
            ctc_loss_gradient(np.zeros((2, 3)), [1, 1])
        with self.assertRaisesRegex(ValueError, "nonblank"):
            ctc_loss_gradient(np.zeros((2, 3)), [0])


if __name__ == "__main__":
    unittest.main()
