"""CLI end-to-end: embed/extract/analyze/capacity over real temp files."""

from __future__ import annotations

import contextlib
import io
import random
import tempfile
import unittest
from pathlib import Path

import cli
from bmp import gradient_image, load_bmp, save_bmp


def _run(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(argv)
    return code, out.getvalue(), err.getvalue()


class CliTests(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.cover = self.dir / "cover.bmp"
        save_bmp(self.cover, gradient_image(64, 64))
        self.msg = self.dir / "msg.bin"
        rng = random.Random(99)
        self.payload = bytes(rng.randrange(256) for _ in range(400))
        self.msg.write_bytes(self.payload)

    def test_embed_extract_roundtrip(self):
        stego = self.dir / "stego.bmp"
        out_msg = self.dir / "out.bin"

        code, out, err = _run([
            "embed", "--in", str(self.cover), "--out", str(stego),
            "--key", "sekrit", "--file", str(self.msg),
        ])
        self.assertEqual(code, 0, err)
        self.assertIn("embedded 400 bytes", out)
        self.assertTrue(stego.exists())

        code, out, err = _run([
            "extract", "--in", str(stego), "--out", str(out_msg),
            "--key", "sekrit",
        ])
        self.assertEqual(code, 0, err)
        self.assertIn("HMAC ok", out)
        self.assertEqual(out_msg.read_bytes(), self.payload)

    def test_multi_plane_roundtrip(self):
        stego = self.dir / "stego2.bmp"
        out_msg = self.dir / "out2.bin"
        code, _, err = _run([
            "embed", "--in", str(self.cover), "--out", str(stego),
            "--key", "k", "--file", str(self.msg), "--planes", "2",
        ])
        self.assertEqual(code, 0, err)
        code, _, err = _run([
            "extract", "--in", str(stego), "--out", str(out_msg),
            "--key", "k", "--planes", "2",
        ])
        self.assertEqual(code, 0, err)
        self.assertEqual(out_msg.read_bytes(), self.payload)

    def test_analyze_clean_image(self):
        code, out, err = _run(["analyze", "--in", str(self.cover)])
        self.assertEqual(code, 0, err)
        self.assertIn("clean", out)
        self.assertIn("chi2=", out)

    def test_analyze_embedded_image(self):
        stego = self.dir / "stego.bmp"
        # fill the image nearly completely with random bits: chi-square
        # detection is a heuristic that partial or patterned embeddings
        # can evade (documented), so the fixture fully randomizes LSBs
        import random as _random

        from stego import capacity_bytes
        from bmp import load_bmp

        cover = load_bmp(self.cover)
        rng = _random.Random(11)
        big = self.dir / "big.bin"
        big.write_bytes(
            bytes(rng.randrange(256)
                  for _ in range(capacity_bytes(cover, 0)))
        )
        _run([
            "embed", "--in", str(self.cover), "--out", str(stego),
            "--key", "k", "--file", str(big),
        ])
        code, out, err = _run(["analyze", "--in", str(stego)])
        self.assertEqual(code, 0, err)
        self.assertIn("SUSPICIOUS", out)

    def test_capacity_command(self):
        code, out, err = _run(["capacity", "--in", str(self.cover)])
        self.assertEqual(code, 0, err)
        self.assertIn("1511 bytes", out)
        code, out, _ = _run(
            ["capacity", "--in", str(self.cover), "--planes", "3"]
        )
        self.assertEqual(code, 0)
        self.assertIn("6119 bytes", out)

    def test_wrong_key_exits_1(self):
        stego = self.dir / "stego.bmp"
        _run([
            "embed", "--in", str(self.cover), "--out", str(stego),
            "--key", "right", "--file", str(self.msg),
        ])
        code, _, err = _run([
            "extract", "--in", str(stego),
            "--out", str(self.dir / "o.bin"), "--key", "wrong",
        ])
        self.assertEqual(code, 1)
        self.assertIn("error:", err)
        self.assertIn("HMAC", err)

    def test_oversized_payload_exits_1(self):
        big = self.dir / "big.bin"
        big.write_bytes(b"x" * 100000)
        code, _, err = _run([
            "embed", "--in", str(self.cover),
            "--out", str(self.dir / "s.bmp"),
            "--key", "k", "--file", str(big),
        ])
        self.assertEqual(code, 1)
        self.assertIn("exceeds capacity", err)

    def test_missing_input_file_exits_1(self):
        code, _, err = _run([
            "analyze", "--in", str(self.dir / "absent.bmp"),
        ])
        self.assertEqual(code, 1)
        self.assertIn("error:", err)

    def test_corrupt_bmp_exits_1(self):
        bad = self.dir / "bad.bmp"
        bad.write_bytes(b"this is not a bitmap at all, really not")
        code, _, err = _run(["analyze", "--in", str(bad)])
        self.assertEqual(code, 1)
        self.assertIn("error:", err)

    def test_extract_from_clean_image_exits_1(self):
        code, _, err = _run([
            "extract", "--in", str(self.cover),
            "--out", str(self.dir / "o.bin"), "--key", "k",
        ])
        self.assertEqual(code, 1)
        self.assertIn("error:", err)

    def test_invalid_planes_value_exits_1(self):
        code, _, err = _run([
            "embed", "--in", str(self.cover),
            "--out", str(self.dir / "s.bmp"),
            "--key", "k", "--file", str(self.msg), "--planes", "7",
        ])
        self.assertEqual(code, 1)
        self.assertIn("error:", err)

    def test_missing_required_argument_exits_2(self):
        with self.assertRaises(SystemExit):
            cli.main(["embed"])

    def test_no_command_exits_2(self):
        with self.assertRaises(SystemExit):
            cli.main([])


if __name__ == "__main__":
    unittest.main()
