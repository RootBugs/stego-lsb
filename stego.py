"""LSB steganography: embed/extract payloads in 24-bit BMP images.

Wire format of the hidden stream (written bit-by-bit, MSB first, across
the selected bit planes of every RGB byte in raster order):

    magic "OSSS" (4) | version/planes (1) | payload length (4, BE)
    | HMAC-SHA256 tag, truncated to 16 bytes (16) | payload (length)

The 25-byte header (200 bits) always occupies the first bits of the
stream.  The HMAC is keyed with the same secret as the payload, so a
wrong key or any tampering is detected *before* the payload is returned
(:class:`StegoError`, never silent garbage).

Bit planes 0..3 are selectable: plane 0 is the LSB of each byte, plane 1
the next bit up, etc.  Capacity grows linearly with the plane count but
so does detectability — plane 0 is the subtlest.
"""

from __future__ import annotations

import hashlib
import hmac
import struct
from pathlib import Path

from bmp import Image, write_bmp

__all__ = [
    "StegoError",
    "MAGIC",
    "HEADER_SIZE",
    "MAX_PLANE",
    "capacity_bits",
    "capacity_bytes",
    "embed",
    "extract",
    "save_embedded",
    "load_and_extract",
]

MAGIC = b"OSSS"
_VERSION = 1
_PLANE_STRUCT = struct.Struct(">B")       # version in high nibble style
_LEN_STRUCT = struct.Struct(">I")         # payload length, big-endian
TAG_SIZE = 16                              # truncated HMAC-SHA256
HEADER_SIZE = len(MAGIC) + 1 + 4 + TAG_SIZE  # 25 bytes = 200 bits
MAX_PLANE = 3                              # planes 0..3


class StegoError(ValueError):
    """Raised on capacity violations, missing headers or HMAC failure."""


def _check_planes(planes: int) -> None:
    if not isinstance(planes, int) or not 0 <= planes <= MAX_PLANE:
        raise StegoError(
            f"planes must be an integer 0..{MAX_PLANE}, got {planes!r}"
        )


def capacity_bits(image: Image, planes: int) -> int:
    """Bits available for the *hidden stream* (header + payload).

    Every RGB byte contributes ``(planes + 1)`` usable bits (one per
    selected bit plane), so plane 0 gives one bit per byte.
    """
    _check_planes(planes)
    return image.byte_count * (planes + 1)


def capacity_bytes(image: Image, planes: int) -> int:
    """Maximum payload bytes for *image* at *planes* (after the header)."""
    bits = capacity_bits(image, planes) - HEADER_SIZE * 8
    return bits // 8 if bits > 0 else 0


def _stream_bits(stream: bytes):
    """Yield bits of *stream* MSB first."""
    for byte in stream:
        for shift in range(7, -1, -1):
            yield (byte >> shift) & 1


def embed(image: Image, payload: bytes, key: bytes, planes: int = 0) -> Image:
    """Return a new Image with *payload* hidden in bit planes of *planes*.

    Raises :class:`StegoError` when the payload (plus header) does not fit
    or arguments are invalid.  The input image is never modified.
    """
    _check_planes(planes)
    if not isinstance(payload, (bytes, bytearray)):
        raise StegoError("payload must be bytes-like")
    payload = bytes(payload)
    if not isinstance(key, (bytes, bytearray)) or len(key) == 0:
        raise StegoError("key must be a non-empty bytes-like secret")
    needed_bits = (HEADER_SIZE + len(payload)) * 8
    if needed_bits > capacity_bits(image, planes):
        raise StegoError(
            f"payload of {len(payload)} bytes exceeds capacity "
            f"{capacity_bytes(image, planes)} bytes at planes={planes}"
        )
    tag = hmac.new(bytes(key), payload, hashlib.sha256).digest()[:TAG_SIZE]
    header = (
        MAGIC
        + _PLANE_STRUCT.pack(_VERSION << 4 | planes)
        + _LEN_STRUCT.pack(len(payload))
        + tag
    )
    stream = header + payload

    out = bytearray(image.data)
    bit_iter = _stream_bits(stream)
    # Walk bytes in raster order; within each byte, low plane first.
    for byte_index in range(len(out)):
        for plane in range(planes + 1):
            bit = next(bit_iter, None)
            if bit is None:
                # Stream fully written (capacity check guarantees fit).
                return Image(image.width, image.height, bytes(out))
            mask = 1 << plane
            out[byte_index] = (out[byte_index] & ~mask) | (bit << plane)
    # Stream exactly fills the image: iterator is now exhausted.
    if next(bit_iter, None) is not None:
        raise RuntimeError("internal error: stream did not fit image")
    return Image(image.width, image.height, bytes(out))


def _read_stream(image: Image, planes: int, byte_count: int) -> bytes:
    """Read the first *byte_count* bytes of the hidden bit stream."""
    out = bytearray()
    bit_buffer = 0
    bit_count = 0
    for byte_index in range(len(image.data)):
        for plane in range(planes + 1):
            bit = (image.data[byte_index] >> plane) & 1
            bit_buffer = (bit_buffer << 1) | bit
            bit_count += 1
            if bit_count == 8:
                out.append(bit_buffer)
                bit_buffer = 0
                bit_count = 0
                if len(out) == byte_count:
                    return bytes(out)
    raise StegoError("image does not contain enough hidden bits")


def extract(image: Image, key: bytes, planes: int = 0) -> bytes:
    """Recover the payload hidden at *planes*; verify its HMAC.

    Raises :class:`StegoError` for a missing/foreign header, an
    inconsistent planes field, a truncated stream, a wrong key or a
    tampered payload — the HMAC is checked with ``hmac.compare_digest``
    and never bypassed.
    """
    _check_planes(planes)
    if not isinstance(key, (bytes, bytearray)) or len(key) == 0:
        raise StegoError("key must be a non-empty bytes-like secret")
    header = _read_stream(image, planes, HEADER_SIZE)
    if header[: len(MAGIC)] != MAGIC:
        raise StegoError(
            "no stego header found (wrong planes or no payload embedded)"
        )
    meta = header[len(MAGIC)]
    version, stored_planes = meta >> 4, meta & 0x0F
    if version != _VERSION:
        raise StegoError(f"unsupported stego format version {version}")
    if stored_planes != planes:
        raise StegoError(
            f"header says planes={stored_planes}, extraction asked "
            f"for planes={planes}"
        )
    payload_len = _LEN_STRUCT.unpack(
        header[len(MAGIC) + 1: len(MAGIC) + 5]
    )[0]
    max_len = capacity_bytes(image, planes)
    if payload_len > max_len:
        raise StegoError(
            f"header claims {payload_len} byte payload but capacity "
            f"is {max_len}"
        )
    expected_bits = (HEADER_SIZE + payload_len) * 8
    if capacity_bits(image, planes) < expected_bits:
        raise StegoError("image too small for declared payload length")
    body = _read_stream(image, planes, HEADER_SIZE + payload_len)
    tag = header[len(MAGIC) + 5: HEADER_SIZE]
    payload = body[HEADER_SIZE:]
    want = hmac.new(bytes(key), payload, hashlib.sha256).digest()[:TAG_SIZE]
    if not hmac.compare_digest(tag, want):
        raise StegoError(
            "HMAC verification failed: wrong key or tampered image"
        )
    return payload


def save_embedded(path, image: Image, payload: bytes, key: bytes,
                  planes: int = 0) -> Image:
    """Embed *payload* and write the resulting BMP to *path*."""
    stego = embed(image, payload, key, planes)
    Path(path).write_bytes(write_bmp(stego))
    return stego


def load_and_extract(path, key: bytes, planes: int = 0) -> bytes:
    """Read a BMP from *path* and extract its payload."""
    from bmp import read_bmp

    return extract(read_bmp(Path(path).read_bytes()), key, planes)
