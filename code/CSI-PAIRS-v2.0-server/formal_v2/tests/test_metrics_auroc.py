import unittest

import numpy as np
from scipy.stats import rankdata

from formal_v2.formal_metrics import binary_auroc


class AurocTests(unittest.TestCase):
    def test_matches_pairwise_definition_with_ties_and_permutations(self):
        rng = np.random.default_rng(932)
        for size in (2, 3, 17, 128):
            labels = np.arange(size) % 2
            for scores in (np.zeros(size), rng.integers(-3, 4, size), rng.normal(size=size)):
                positive = scores[labels == 1, None]
                negative = scores[labels == 0][None, :]
                expected = np.mean((positive > negative) + 0.5 * (positive == negative))
                for _ in range(3):
                    order = rng.permutation(size)
                    self.assertEqual(binary_auroc(labels[order], scores[order]), expected)

    def test_large_mixed_ties_match_rank_reference(self):
        rng = np.random.default_rng(93)
        labels = np.arange(100_000) % 2
        scores = np.round(rng.normal(size=labels.size), 4)
        ranks = rankdata(scores, method="average")
        count = int(labels.sum())
        expected = (ranks[labels == 1].sum() - count * (count + 1) / 2) / (count * (labels.size - count))
        self.assertEqual(binary_auroc(labels, scores), expected)

    def test_validation_is_preserved(self):
        for labels, scores in (
            ([0, 0], [1, 2]), ([0, 2], [1, 2]),
            ([0, 1], [1, np.nan]), ([0, 1], [1, np.inf]),
            ([0, 1], [1]), ([], []),
        ):
            with self.assertRaises(ValueError):
                binary_auroc(np.asarray(labels), np.asarray(scores))
