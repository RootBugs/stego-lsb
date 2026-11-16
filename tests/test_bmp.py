"""BMP writer/reader: roundtrip, bottom-up layout, padding, strictness."""

from __future__ import annotations

import struct
import unittest

from bmp import (
    PIXEL_OFFSET,
    BmpError,
    Image,
    gradient_image,
    random_image,
    read_bmp,
    row_stride,
    save_bmp,
    load_bmp,
    write_bmp,
)


class ImageTests(unittest.TestCase):

    def test_rejects_non_positive_dimensions(self):
        with self.assertRaises(BmpError):
            Image(0, 10)
        with self.assertRaises(BmpError):
            Image(10, -1)

    def test_rejects_wrong_data_length(self):
        with self.assertRaises(BmpError):
            Image(4, 4, b"too short")

    def test_default_data_is_black(self):
        img = Image(3, 2)
        self.assertEqual(img.data, bytes(18))
        self.assertEqual(img.pixel_count, 6)
        self.assertEqual(img.byte_count, 18)

    def test_equality(self):
        a = gradient_image(8, 8)
        b = gradient_image(8, 8)
        self.assertEqual(a, b)
        self.assertNotEqual(a, random_image(8, 8, seed=1))


class BmpRoundtripTests(unittest.TestCase):

    def test_write_read_roundtrip_byte_exact(self):
        for w, h in ((1, 1), (7, 5), (16, 16), (33, 3)):
            img = random_image(w, h, seed=w * 100 + h)
            self.assertEqual(read_bmp(write_bmp(img)), img, f"{w}x{h}")

    def test_file_starts_with_bm_magic(self):
        blob = write_bmp(gradient_image(4, 4))
        self.assertEqual(blob[:2], b"BM")

    def test_pixel_offset_is_54(self):
        blob = write_bmp(gradient_image(4, 4))
        offset = struct.unpack_from("<I", blob, 10)[0]
        self.assertEqual(offset, PIXEL_OFFSET)
        self.assertEqual(PIXEL_OFFSET, 54)

    def test_row_padding_to_four_bytes(self):
        # width=5 -> 15 bytes/row -> stride 16 (1 pad byte per row)
        self.assertEqual(row_stride(5), 16)
        self.assertEqual(row_stride(4), 12)
        self.assertEqual(row_stride(1), 4)
        blob = write_bmp(gradient_image(5, 2))
        stride = row_stride(5)
        self.assertEqual(len(blob), 54 + stride * 2)
        # padding bytes at the end of each stored row are zero
        for row in range(2):
            pad_start = 54 + row * stride + 15
            self.assertEqual(blob[pad_start:pad_start + 1], b"\x00")

    def test_rows_are_stored_bottom_up_bgr(self):
        # 2x2 image with distinct pixels; verify on-disk byte order.
        data = bytes([
            0x11, 0x22, 0x33,   # top-left    R=11 G=22 B=33
            0x44, 0x55, 0x66,   # top-right
            0x77, 0x88, 0x99,   # bottom-left
            0xAA, 0xBB, 0xCC,   # bottom-right
        ])
        img = Image(2, 2, data)
        blob = write_bmp(img)
        stride = row_stride(2)  # 6 pixel bytes -> 8 with padding
        self.assertEqual(stride, 8)
        # first stored row is the BOTTOM row, in BGR order
        first_pixel = blob[54:57]
        self.assertEqual(first_pixel, bytes((0x99, 0x88, 0x77)))
        # second stored row starts after stride bytes: the TOP row
        second_pixel = blob[54 + stride:54 + stride + 3]
        self.assertEqual(second_pixel, bytes((0x33, 0x22, 0x11)))

    def test_roundtrip_preserves_pixel_positions(self):
        data = bytes(range(18))  # 3x2
        img = Image(3, 2, data)
        back = read_bmp(write_bmp(img))
        self.assertEqual(back.data, data)
        # top-left pixel specifically
        self.assertEqual(back.data[:3], data[:3])


class BmpStrictnessTests(unittest.TestCase):

    def test_rejects_truncated_header(self):
        with self.assertRaises(BmpError):
            read_bmp(b"BM" + bytes(10))

    def test_rejects_bad_magic(self):
        blob = bytearray(write_bmp(gradient_image(4, 4)))
        blob[0:2] = b"XX"
        with self.assertRaises(BmpError):
            read_bmp(bytes(blob))

    def test_rejects_24_bit_requirement(self):
        blob = bytearray(write_bmp(gradient_image(4, 4)))
        struct.pack_into("<H", blob, 28, 32)  # biBitCount = 32
        with self.assertRaises(BmpError):
            read_bmp(bytes(blob))

    def test_rejects_compressed(self):
        blob = bytearray(write_bmp(gradient_image(4, 4)))
        struct.pack_into("<I", blob, 30, 1)  # biCompression = BI_RLE8
        with self.assertRaises(BmpError):
            read_bmp(bytes(blob))

    def test_rejects_top_down(self):
        blob = bytearray(write_bmp(gradient_image(4, 4)))
        struct.pack_into("<i", blob, 22, -4)  # negative height
        with self.assertRaises(BmpError):
            read_bmp(bytes(blob))

    def test_rejects_truncated_pixel_data(self):
        blob = write_bmp(gradient_image(8, 8))
        with self.assertRaises(BmpError):
            read_bmp(blob[:-5])

    def test_rejects_non_bytes_input(self):
        with self.assertRaises(BmpError):
            read_bmp("not bytes")

    def test_save_load_via_path(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.bmp"
            img = gradient_image(9, 6)
            save_bmp(path, img)
            self.assertEqual(load_bmp(path), img)


if __name__ == "__main__":
    unittest.main()
