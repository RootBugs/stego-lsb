"""Embed/extract: roundtrips, capacity, HMAC failures, header behavior."""

from __future__ import annotations

import random
import unittest

from bmp import gradient_image, random_image, read_bmp, write_bmp
from stego import (
    HEADER_SIZE,
    MAGIC,
    StegoError,
    capacity_bits,
    capacity_bytes,
    embed,
    extract,
    load_and_extract,
    save_embedded,
)

KEY = b"correct horse battery staple"


def _payload(size: int, seed: int = 42) -> bytes:
    rng = random.Random(seed)
    return bytes(rng.randrange(256) for _ in range(size))


class CapacityTests(unittest.TestCase):

    def setUp(self):
        self.img = gradient_image(64, 64)  # 4096 pixels * 3 bytes

    def test_capacity_formula_single_plane(self):
        # 12288 bytes * 1 bit (plane 0) - 200 header bits, floored to bytes
        expected = (12288 - HEADER_SIZE * 8) // 8
        self.assertEqual(capacity_bytes(self.img, 0), expected)
        self.assertEqual(capacity_bytes(self.img, 0), 1511)

    def test_multi_plane_capacity_math(self):
        # each additional plane adds one bit per byte
        for planes, mult in ((0, 1), (1, 2), (2, 3), (3, 4)):
            bits = capacity_bits(self.img, planes)
            self.assertEqual(bits, 12288 * mult)
            self.assertEqual(
                capacity_bytes(self.img, planes),
                (12288 * mult - HEADER_SIZE * 8) // 8,
            )
        # linear growth: plane 3 == 4x plane 0 bits
        self.assertEqual(
            capacity_bits(self.img, 3), 4 * capacity_bits(self.img, 0)
        )

    def test_capacity_grows_with_image(self):
        small = gradient_image(8, 8)
        self.assertLess(capacity_bytes(small, 0),
                        capacity_bytes(self.img, 0))

    def test_tiny_image_has_zero_or_no_capacity(self):
        tiny = gradient_image(1, 1)  # 24 bits < 200-bit header
        self.assertEqual(capacity_bytes(tiny, 0), 0)
        with self.assertRaises(StegoError):
            embed(tiny, b"", KEY, 0)  # even empty payload needs header

    def test_invalid_planes_rejected(self):
        for bad in (-1, 4, 10, "0", None):
            with self.assertRaises(StegoError):
                capacity_bytes(self.img, bad)
            with self.assertRaises(StegoError):
                embed(self.img, b"x", KEY, bad)


class RoundtripTests(unittest.TestCase):

    def setUp(self):
        self.img = gradient_image(64, 64)

    def test_roundtrip_byte_exact(self):
        payload = _payload(500)
        stego = embed(self.img, payload, KEY, planes=0)
        self.assertEqual(extract(stego, KEY, planes=0), payload)

    def test_roundtrip_various_sizes(self):
        cap = capacity_bytes(self.img, 0)
        for size in (0, 1, 2, 17, 255, 1000, cap):
            payload = _payload(size, seed=size)
            stego = embed(self.img, payload, KEY, planes=0)
            self.assertEqual(
                extract(stego, KEY, planes=0), payload, f"size={size}"
            )

    def test_roundtrip_all_plane_settings(self):
        payload = _payload(300, seed=9)
        for planes in (0, 1, 2, 3):
            stego = embed(self.img, payload, KEY, planes=planes)
            self.assertEqual(extract(stego, KEY, planes=planes), payload)
            # extraction at the wrong plane count must not succeed
            for other in (0, 1, 2, 3):
                if other == planes:
                    continue
                with self.assertRaises(StegoError):
                    extract(stego, KEY, planes=other)

    def test_roundtrip_after_bmp_file_cycle(self):
        payload = _payload(400, seed=77)
        stego = embed(self.img, payload, KEY, planes=1)
        reloaded = read_bmp(write_bmp(stego))
        self.assertEqual(extract(reloaded, KEY, planes=1), payload)

    def test_roundtrip_on_random_cover(self):
        cover = random_image(48, 48, seed=1234)
        payload = _payload(700, seed=5)
        stego = embed(cover, payload, KEY, planes=0)
        self.assertEqual(extract(stego, KEY, planes=0), payload)

    def test_embed_does_not_mutate_cover(self):
        before = bytes(self.img.data)
        embed(self.img, _payload(100), KEY, planes=0)
        self.assertEqual(self.img.data, before)

    def test_stego_differs_from_cover(self):
        payload = _payload(100)
        stego = embed(self.img, payload, KEY, planes=0)
        self.assertNotEqual(stego.data, self.img.data)
        # dimensions preserved
        self.assertEqual(
            (stego.width, stego.height), (self.img.width, self.img.height)
        )


class OversizedPayloadTests(unittest.TestCase):

    def setUp(self):
        self.img = gradient_image(64, 64)

    def test_oversized_payload_rejected(self):
        cap = capacity_bytes(self.img, 0)
        with self.assertRaises(StegoError) as ctx:
            embed(self.img, b"x" * (cap + 1), KEY, planes=0)
        self.assertIn("exceeds capacity", str(ctx.exception))

    def test_exactly_capacity_fits(self):
        cap = capacity_bytes(self.img, 0)
        stego = embed(self.img, b"y" * cap, KEY, planes=0)
        self.assertEqual(extract(stego, KEY, planes=0), b"y" * cap)

    def test_oversized_at_higher_plane_still_checked(self):
        payload = _payload(capacity_bytes(self.img, 1) + 1, seed=3)
        with self.assertRaises(StegoError):
            embed(self.img, payload, KEY, planes=1)

    def test_non_bytes_payload_rejected(self):
        with self.assertRaises(StegoError):
            embed(self.img, "not bytes", KEY, planes=0)

    def test_empty_or_missing_key_rejected(self):
        with self.assertRaises(StegoError):
            embed(self.img, b"data", b"", planes=0)
        with self.assertRaises(StegoError):
            extract(embed(self.img, b"data", KEY), b"")


class IntegrityFailureTests(unittest.TestCase):

    def setUp(self):
        self.img = gradient_image(64, 64)
        self.payload = _payload(400, seed=11)
        self.stego = embed(self.img, self.payload, KEY, planes=0)

    def test_wrong_key_fails_cleanly(self):
        with self.assertRaises(StegoError) as ctx:
            extract(self.stego, b"totally different key", planes=0)
        self.assertIn("HMAC", str(ctx.exception))

    def test_wrong_key_error_is_not_garbage_output(self):
        # must raise, never return partially-decoded bytes
        try:
            result = extract(self.stego, b"nope", planes=0)
        except StegoError:
            return
        self.fail(f"wrong key returned payload instead: {result!r}")

    def test_single_bit_tamper_detected(self):
        data = bytearray(self.stego.data)
        # at planes=0 the stream occupies the LSB of the first
        # (header+payload)*8 image bytes; byte 1000 is payload region
        data[1000] ^= 0x01
        tampered = type(self.stego)(self.stego.width, self.stego.height,
                                    bytes(data))
        with self.assertRaises(StegoError):
            extract(tampered, KEY, planes=0)

    def test_tamper_in_header_detected(self):
        data = bytearray(self.stego.data)
        # image byte 45 holds stream bit 45 -> inside the 4-byte length
        # field (stream bits 40..71): corrupts the declared length
        data[45] ^= 0x01
        tampered = type(self.stego)(self.stego.width, self.stego.height,
                                    bytes(data))
        with self.assertRaises(StegoError):
            extract(tampered, KEY, planes=0)

    def test_extract_from_clean_image_fails(self):
        with self.assertRaises(StegoError) as ctx:
            extract(self.img, KEY, planes=0)
        self.assertIn("header", str(ctx.exception).lower())

    def test_compare_digest_used_for_tag(self):
        # functional check: correct key passes, any other key fails,
        # regardless of payload length (HMAC covers everything)
        for size in (0, 1, 64):
            stego = embed(self.img, _payload(size, seed=size), KEY)
            self.assertEqual(extract(stego, KEY), _payload(size, seed=size))
            with self.assertRaises(StegoError):
                extract(stego, KEY + b"x")


class FileApiTests(unittest.TestCase):

    def test_save_and_load_and_extract(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            cover = Path(tmp) / "cover.bmp"
            stego_path = Path(tmp) / "stego.bmp"
            msg = Path(tmp) / "msg.bin"
            payload = _payload(250, seed=21)
            msg.write_bytes(payload)
            from bmp import save_bmp

            save_bmp(cover, gradient_image(64, 64))
            from bmp import load_bmp

            save_embedded(stego_path, load_bmp(cover), payload, KEY, 0)
            self.assertEqual(
                load_and_extract(stego_path, KEY, 0), payload
            )


if __name__ == "__main__":
    unittest.main()
