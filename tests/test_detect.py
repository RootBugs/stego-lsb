"""Chi-square LSB detection: flags embedded, passes clean, deterministic."""

from __future__ import annotations

import random
import unittest

from bmp import Image, gradient_image, random_image
from detect import (
    CHI_THRESHOLD,
    Detection,
    chi_square,
    chi_square_bytes,
    detect,
    format_report,
    lsb_histogram,
    lsb_histogram_bytes,
)
from stego import embed

KEY = b"detection key"


def _payload(size: int, seed: int = 42) -> bytes:
    rng = random.Random(seed)
    return bytes(rng.randrange(256) for _ in range(size))


class HistogramTests(unittest.TestCase):

    def test_histogram_counts_bytes(self):
        counts = lsb_histogram_bytes(bytes([0, 0, 1, 255, 1]))
        self.assertEqual(counts[0], 2)
        self.assertEqual(counts[1], 2)
        self.assertEqual(counts[255], 1)
        self.assertEqual(sum(counts), 5)

    def test_histogram_of_image_matches_data(self):
        img = gradient_image(8, 8)
        self.assertEqual(lsb_histogram(img), lsb_histogram_bytes(img.data))

    def test_gradient_has_no_odd_bytes(self):
        counts = lsb_histogram(gradient_image(16, 16))
        odd_total = sum(counts[1::2])
        self.assertEqual(odd_total, 0)


class ChiSquareMathTests(unittest.TestCase):

    def test_balanced_pairs_score_near_one(self):
        # 500 even / 500 odd in a single pair: chi2 ~= 0 + tiny float noise
        data = bytes([2] * 500 + [3] * 500)
        det = chi_square_bytes(data)
        self.assertAlmostEqual(det.chi2, 0.0, places=9)
        self.assertAlmostEqual(det.ratio, 0.0, places=9)
        self.assertTrue(det.suspicious)

    def test_single_pair_computes_exact_value(self):
        # pair (2,3): 10 even, 6 odd -> expected 8 each
        # chi2 = (6-8)^2/8 + (10-8)^2/8 = 0.5 + 0.5 = 1.0
        det = chi_square_bytes(bytes([2] * 10 + [3] * 6))
        self.assertAlmostEqual(det.chi2, 1.0, places=12)
        self.assertEqual(det.pairs, 1)

    def test_fully_biased_pair_scores_its_byte_count(self):
        # all-even pair: chi2 = (0-n/2)^2/(n/2) * 2 = n
        det = chi_square_bytes(bytes([4] * 50))
        self.assertAlmostEqual(det.chi2, 50.0, places=9)
        self.assertEqual(det.pairs, 1)

    def test_zero_count_pairs_are_skipped(self):
        det = chi_square_bytes(bytes([10] * 20))  # only one pair active
        self.assertEqual(det.pairs, 1)

    def test_empty_input_is_not_suspicious(self):
        det = chi_square_bytes(b"")
        self.assertEqual(det.pairs, 0)
        self.assertFalse(det.suspicious)


class DetectionBehaviorTests(unittest.TestCase):

    def test_clean_gradient_passes(self):
        det = detect(gradient_image(64, 64))
        self.assertFalse(det.suspicious)
        self.assertGreater(det.ratio, CHI_THRESHOLD)
        # every byte even -> chi2 equals the byte count
        self.assertAlmostEqual(det.chi2, 64 * 64 * 3, places=6)

    def test_fully_embedded_image_is_flagged(self):
        cover = gradient_image(64, 64)
        payload = _payload(_capacity(cover), seed=7)
        stego = embed(cover, payload, KEY, planes=0)
        det = detect(stego)
        self.assertTrue(det.suspicious)
        self.assertLess(det.ratio, CHI_THRESHOLD)
        self.assertGreater(det.chi2, 0.0)

    def test_detection_is_deterministic(self):
        cover = gradient_image(64, 64)
        stego = embed(cover, _payload(_capacity(cover), seed=7), KEY, 0)
        first = detect(stego)
        second = detect(stego)
        self.assertEqual(first, second)
        # repeated fresh objects, same values
        self.assertEqual(detect(cover), detect(cover))

    def test_threshold_sits_between_fixtures(self):
        clean = detect(gradient_image(64, 64))
        stego = embed(gradient_image(64, 64),
                      _payload(_capacity(gradient_image(64, 64)), seed=7),
                      KEY, 0)
        dirty = detect(stego)
        self.assertLess(dirty.ratio, CHI_THRESHOLD)
        self.assertGreater(clean.ratio, CHI_THRESHOLD)
        # wide separation margin, no borderline flake
        self.assertGreater(clean.ratio, 10 * CHI_THRESHOLD)

    def test_random_cover_also_measurable(self):
        # random cover: LSBs already balanced -> flagged (documented
        # heuristic limitation, asserted here as defined behavior)
        det = detect(random_image(32, 32, seed=5))
        self.assertIsInstance(det, Detection)
        self.assertLess(det.ratio, CHI_THRESHOLD)
        self.assertTrue(det.suspicious)

    def test_single_plane_embed_changes_ratio_dramatically(self):
        cover = gradient_image(64, 64)
        before = detect(cover).ratio
        after = detect(embed(cover, _payload(_capacity(cover), seed=3),
                             KEY, 0)).ratio
        self.assertGreater(before, 50.0)
        self.assertLess(after, 4.0)

    def test_custom_threshold_respected(self):
        cover = gradient_image(64, 64)
        # absurdly high threshold makes even the clean image suspicious
        self.assertTrue(detect(cover, threshold=1e9).suspicious)
        # absurdly low threshold makes the embedded image pass
        stego = embed(cover, _payload(_capacity(cover), seed=7), KEY, 0)
        self.assertFalse(detect(stego, threshold=1e-12).suspicious)

    def test_format_report_mentions_verdict(self):
        clean = format_report(detect(gradient_image(32, 32)))
        self.assertIn("clean", clean)
        self.assertIn("chi2=", clean)
        stego = embed(gradient_image(64, 64),
                      _payload(_capacity(gradient_image(64, 64)), seed=7),
                      KEY, 0)
        self.assertIn("SUSPICIOUS", format_report(detect(stego)))


def _capacity(img) -> int:
    from stego import capacity_bytes

    return capacity_bytes(img, 0)


if __name__ == "__main__":
    unittest.main()
