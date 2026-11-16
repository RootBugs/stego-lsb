"""Minimal 24-bit uncompressed BMP reader/writer.

Implements exactly the subset needed for LSB steganography:

* ``BITMAPFILEHEADER`` (14 bytes) + ``BITMAPINFOHEADER`` (40 bytes),
  magic ``BM``, ``biBitCount = 24``, ``biCompression = BI_RGB`` (0);
* rows stored **bottom-up** (positive ``biHeight``), each row padded to a
  4-byte boundary with zero bytes;
* pixels stored per row as **BGR** triples (the BMP convention), while
  :class:`Image` keeps a top-down **RGB** buffer for stego logic.

The reader is strict: wrong magic, non-24-bit, compressed, top-down or
truncated files raise :class:`BmpError` instead of guessing.
"""

from __future__ import annotations

import random
import struct
from pathlib import Path

__all__ = [
    "BmpError",
    "Image",
    "row_stride",
    "write_bmp",
    "read_bmp",
    "save_bmp",
    "load_bmp",
    "gradient_image",
    "random_image",
]

_FILE_HEADER = struct.Struct("<2sIHHI")   # 14 bytes
_DIB_HEADER = struct.Struct("<IiiHHIIiiII")  # 40 bytes
FILE_HEADER_SIZE = 14
DIB_HEADER_SIZE = 40
PIXEL_OFFSET = FILE_HEADER_SIZE + DIB_HEADER_SIZE  # 54


class BmpError(ValueError):
    """Raised for any malformed or unsupported BMP input."""


class Image:
    """A top-down 24-bit RGB image: ``data`` is ``width * height * 3``
    bytes, three bytes (R, G, B) per pixel, row-major."""

    __slots__ = ("width", "height", "data")

    def __init__(self, width: int, height: int, data=None) -> None:
        if width <= 0 or height <= 0:
            raise BmpError(
                f"image dimensions must be positive, got {width}x{height}"
            )
        expected = width * height * 3
        if data is None:
            data = bytes(expected)
        elif not isinstance(data, (bytes, bytearray)):
            raise BmpError("image data must be bytes-like")
        else:
            data = bytes(data)
        if len(data) != expected:
            raise BmpError(
                f"image data must be {expected} bytes for "
                f"{width}x{height}, got {len(data)}"
            )
        self.width = width
        self.height = height
        self.data = data

    @property
    def pixel_count(self) -> int:
        return self.width * self.height

    @property
    def byte_count(self) -> int:
        return self.pixel_count * 3

    def __eq__(self, other) -> bool:
        if not isinstance(other, Image):
            return NotImplemented
        return (
            self.width == other.width
            and self.height == other.height
            and self.data == other.data
        )

    def __repr__(self) -> str:
        return f"Image({self.width}x{self.height}, {self.byte_count} bytes)"


def row_stride(width: int) -> int:
    """Bytes per stored BMP row: 3 bytes/pixel rounded up to a multiple
    of 4."""
    return (width * 3 + 3) // 4 * 4


def write_bmp(image: Image) -> bytes:
    """Serialize *image* as a 24-bit bottom-up BMP file."""
    stride = row_stride(image.width)
    image_size = stride * image.height
    file_size = PIXEL_OFFSET + image_size
    file_header = _FILE_HEADER.pack(b"BM", file_size, 0, 0, PIXEL_OFFSET)
    dib = _DIB_HEADER.pack(
        DIB_HEADER_SIZE, image.width, image.height, 1, 24, 0,
        image_size, 2835, 2835, 0, 0,
    )
    padding = bytes(stride - image.width * 3)
    rows = []
    for y in range(image.height - 1, -1, -1):  # bottom-up
        base = y * image.width * 3
        row = bytearray()
        for x in range(image.width):
            i = base + x * 3
            r, g, b = image.data[i], image.data[i + 1], image.data[i + 2]
            row += bytes((b, g, r))  # BGR on disk
        row += padding
        rows.append(bytes(row))
    return file_header + dib + b"".join(rows)


def read_bmp(blob: bytes) -> Image:
    """Parse a 24-bit bottom-up BMP into a top-down RGB :class:`Image`."""
    if not isinstance(blob, (bytes, bytearray)):
        raise BmpError("BMP input must be bytes-like")
    blob = bytes(blob)
    if len(blob) < PIXEL_OFFSET:
        raise BmpError(
            f"truncated BMP: {len(blob)} bytes < {PIXEL_OFFSET}-byte header"
        )
    magic, _file_size, _res1, _res2, offset = _FILE_HEADER.unpack_from(blob, 0)
    if magic != b"BM":
        raise BmpError(f"bad BMP magic {magic!r}, expected b'BM'")
    (
        dib_size, width, height, planes, bpp, compression,
        _img_size, _xppm, _yppm, _colors, _important,
    ) = _DIB_HEADER.unpack_from(blob, FILE_HEADER_SIZE)
    if dib_size < DIB_HEADER_SIZE:
        raise BmpError(f"unsupported DIB header size {dib_size}")
    if height <= 0:
        raise BmpError(
            f"only bottom-up BMPs supported (biHeight={height})"
        )
    if width <= 0:
        raise BmpError(f"invalid BMP width {width}")
    if planes != 1:
        raise BmpError(f"invalid BMP plane count {planes}")
    if bpp != 24:
        raise BmpError(f"only 24-bit BMPs supported (biBitCount={bpp})")
    if compression != 0:
        raise BmpError(
            f"compressed BMPs not supported (biCompression={compression})"
        )
    if offset < PIXEL_OFFSET or offset > len(blob):
        raise BmpError(f"invalid pixel data offset {offset}")
    stride = row_stride(width)
    needed = offset + stride * height
    if len(blob) < needed:
        raise BmpError(
            f"truncated BMP pixel data: need {needed} bytes, "
            f"file has {len(blob)}"
        )
    data = bytearray(width * height * 3)
    for row in range(height):
        # stored bottom-up: file row 0 is the image's bottom row
        y = height - 1 - row
        src = offset + row * stride
        dst = y * width * 3
        for x in range(width):
            i = src + x * 3
            b, g, r = blob[i], blob[i + 1], blob[i + 2]
            j = dst + x * 3
            data[j] = r
            data[j + 1] = g
            data[j + 2] = b
    return Image(width, height, bytes(data))


def save_bmp(path, image: Image) -> None:
    """Write *image* to *path* as a 24-bit BMP."""
    Path(path).write_bytes(write_bmp(image))


def load_bmp(path) -> Image:
    """Read a 24-bit BMP from *path*."""
    return read_bmp(Path(path).read_bytes())


def gradient_image(width: int, height: int) -> Image:
    """Deterministic cover image whose every channel byte is **even**
    (strongly biased LSBs — the natural-image property chi-square
    detection relies on).

    * R = (2x) mod 256, G = (2y) mod 256, B = (2(x+y)) mod 256.
    """
    data = bytearray(width * height * 3)
    i = 0
    for y in range(height):
        for x in range(width):
            data[i] = (2 * x) % 256
            data[i + 1] = (2 * y) % 256
            data[i + 2] = (2 * (x + y)) % 256
            i += 3
    return Image(width, height, bytes(data))


def random_image(width: int, height: int, seed: int = 0) -> Image:
    """Seeded random-pixel image (deterministic test fixtures)."""
    rng = random.Random(seed)
    data = bytes(rng.randrange(256) for _ in range(width * height * 3))
    return Image(width, height, data)
