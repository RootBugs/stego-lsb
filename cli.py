"""CLI: embed / extract / analyze / capacity for 24-bit BMP images.

Usage::

    python cli.py embed   --in cover.bmp --out stego.bmp --key K --file msg.txt [--planes N]
    python cli.py extract --in stego.bmp --key K --out msg.txt [--planes N]
    python cli.py analyze --in image.bmp
    python cli.py capacity --in image.bmp [--planes N]

Exit codes: 0 success, 1 operational error (bad image, capacity,
HMAC failure), 2 usage error (argparse).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bmp import BmpError, load_bmp, save_bmp
from detect import format_report, detect
from stego import StegoError, capacity_bytes, embed, extract

__all__ = ["main"]


def _cmd_embed(args) -> int:
    image = load_bmp(args.input)
    payload = Path(args.file).read_bytes()
    cap = capacity_bytes(image, args.planes)
    if len(payload) > cap:
        print(
            f"error: payload {len(payload)} bytes exceeds capacity "
            f"{cap} bytes (planes={args.planes})",
            file=sys.stderr,
        )
        return 1
    stego = embed(image, payload, args.key.encode("utf-8"), args.planes)
    save_bmp(args.output, stego)
    print(
        f"embedded {len(payload)} bytes -> {args.output} "
        f"(capacity {cap} bytes, planes={args.planes})"
    )
    return 0


def _cmd_extract(args) -> int:
    image = load_bmp(args.input)
    payload = extract(image, args.key.encode("utf-8"), args.planes)
    Path(args.output).write_bytes(payload)
    print(f"extracted {len(payload)} bytes -> {args.output} (HMAC ok)")
    return 0


def _cmd_analyze(args) -> int:
    image = load_bmp(args.input)
    result = detect(image)
    print(f"{args.input}: {image.width}x{image.height}")
    print(format_report(result))
    return 0


def _cmd_capacity(args) -> int:
    image = load_bmp(args.input)
    print(
        f"{args.input}: {image.width}x{image.height}, planes={args.planes} "
        f"-> {capacity_bytes(image, args.planes)} bytes"
    )
    return 0


def _add_planes(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--planes", type=int, default=0, metavar="N",
        help="bit planes to use, 0..3 (default 0 = LSB only)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stego-lsb",
        description="Hide data in 24-bit BMP images — and detect it.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("embed", help="embed a file into a BMP")
    p.add_argument("--in", dest="input", required=True, metavar="BMP")
    p.add_argument("--out", dest="output", required=True, metavar="BMP")
    p.add_argument("--key", required=True, metavar="SECRET")
    p.add_argument("--file", required=True, metavar="PAYLOAD")
    _add_planes(p)
    p.set_defaults(func=_cmd_embed)

    p = sub.add_parser("extract", help="extract a payload from a BMP")
    p.add_argument("--in", dest="input", required=True, metavar="BMP")
    p.add_argument("--out", dest="output", required=True, metavar="FILE")
    p.add_argument("--key", required=True, metavar="SECRET")
    _add_planes(p)
    p.set_defaults(func=_cmd_extract)

    p = sub.add_parser("analyze", help="chi-square LSB detection")
    p.add_argument("--in", dest="input", required=True, metavar="BMP")
    p.set_defaults(func=_cmd_analyze)

    p = sub.add_parser("capacity", help="report payload capacity")
    p.add_argument("--in", dest="input", required=True, metavar="BMP")
    _add_planes(p)
    p.set_defaults(func=_cmd_capacity)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (BmpError, StegoError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
