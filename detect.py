"""Chi-square detection of LSB steganography.

Westfeld-style chi-square analysis over the LSB histogram.  For each
even/odd value pair ``(2i, 2i+1)`` of the byte histogram, a *clean*
image with natural (biased) LSBs — e.g. every byte even — sits far from
the 50/50 split, while payload bits (effectively random) push each pair
toward balance.  The per-pair statistic is

    chi2 = sum over active pairs of
           (observed - expected)^2 / expected   (both cells)

and we report it **normalized by the number of active pairs**
(``ratio = chi2 / pairs``): a balanced image scores ~1.0 (chi-square with
one degree of freedom per pair), a biased clean image scores much higher
(since ``chi2 == total byte count`` when one cell is empty).

An image is flagged ``suspicious`` when ``ratio < CHI_THRESHOLD``.
The threshold is a fixed constant — deterministic on the synthetic
fixtures used in tests (the test path contains no randomness of its
own).

Scope and honesty: the analysis inspects plane 0 (the LSB) of every RGB
byte.  It is a *heuristic detector*: partially embedded payloads leave
enough biased bytes to evade it, and cover images that already have
uniform LSBs (noise, already-encrypted content) false-positive.  See
README.
"""

from __future__ import annotations

from typing import NamedTuple

from bmp import Image

__all__ = [
    "CHI_THRESHOLD",
    "Detection",
    "lsb_histogram",
    "lsb_histogram_bytes",
    "chi_square_bytes",
    "chi_square",
    "detect",
    "format_report",
]

# Fixed threshold: balanced (embedded) images score ~1, clean synthetic
# covers score >= ~9 on every fixture used here.  Chosen for
# determinism, not fitted to any external corpus.
CHI_THRESHOLD = 4.0


class Detection(NamedTuple):
    """Result of :func:`detect`."""

    chi2: float          # raw statistic
    ratio: float         # chi2 / active pairs (~1.0 == balanced LSBs)
    suspicious: bool     # ratio < threshold
    threshold: float
    pairs: int           # even/odd pairs with at least one observation
    odd_ratio: float     # observed odd / expected odd (1.0 == balanced)


def lsb_histogram_bytes(data: bytes) -> list:
    """Counts of byte values 0..255 over *data*."""
    counts = [0] * 256
    for byte in data:
        counts[byte] += 1
    return counts


def lsb_histogram(image: Image) -> list:
    """Counts of RGB byte values 0..255 in *image*."""
    return lsb_histogram_bytes(image.data)


def chi_square_bytes(data: bytes,
                     threshold: float = CHI_THRESHOLD) -> Detection:
    """Chi-square LSB analysis over a raw byte buffer."""
    counts = lsb_histogram_bytes(data)
    chi2 = 0.0
    pairs = 0
    observed_odd = 0
    expected_odd = 0.0
    for i in range(128):
        even = counts[2 * i]
        odd = counts[2 * i + 1]
        total = even + odd
        if total == 0:
            continue
        expected = total / 2.0
        chi2 += ((odd - expected) ** 2) / expected
        chi2 += ((even - expected) ** 2) / expected
        pairs += 1
        observed_odd += odd
        expected_odd += expected
    ratio = chi2 / pairs if pairs else 0.0
    odd_ratio = (observed_odd / expected_odd) if expected_odd else 0.0
    return Detection(
        chi2=chi2,
        ratio=ratio,
        suspicious=pairs > 0 and ratio < threshold,
        threshold=threshold,
        pairs=pairs,
        odd_ratio=odd_ratio,
    )


def chi_square(image: Image,
               threshold: float = CHI_THRESHOLD) -> Detection:
    """Chi-square LSB analysis of an *image* (plane 0 of every byte)."""
    return chi_square_bytes(image.data, threshold)


def detect(image: Image, threshold: float = CHI_THRESHOLD) -> Detection:
    """Alias of :func:`chi_square` — report whether *image* looks
    stego-suspicious."""
    return chi_square(image, threshold)


def format_report(detection: Detection) -> str:
    """Human-readable report block used by the CLI."""
    verdict = (
        "SUSPICIOUS (likely embedded)" if detection.suspicious
        else "clean (no signal)"
    )
    return (
        f"chi2={detection.chi2:.3f}  ratio={detection.ratio:.4f}  "
        f"threshold={detection.threshold:.2f}  pairs={detection.pairs}  "
        f"odd/even={detection.odd_ratio:.4f}\n"
        f"verdict: {verdict}"
    )
