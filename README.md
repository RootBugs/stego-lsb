# stego-lsb — Hide Data in Plain Sight, and Detect It

LSB steganography for 24-bit BMP images, in pure Python: embed a payload
across selectable **bit planes 0–3**, protect it with a **keyed
HMAC-SHA256**, and run **chi-square detection** to see whether an image
is hiding something. Includes its own minimal BMP writer/reader — no
Pillow, no dependencies.

## Why it's special

- **Own BMP implementation**: 54-byte headers, bottom-up rows, BGR pixel
  order, 4-byte row padding — written and parsed byte-exactly, with a
  strict reader that rejects compressed/top-down/32-bit/truncated files.
- **Integrity, not hope**: a 25-byte header (`OSSS` magic ‖ version/planes
  ‖ length ‖ 16-byte HMAC tag) precedes every payload. Wrong key or a
  single flipped bit → clean `StegoError`, never garbage output
  (verification uses `hmac.compare_digest`).
- **Capacity before embed**: `capacity_bytes(image, planes)` tells you
  exactly how much fits (`byte_count × (planes+1) − 200 header bits`);
  oversized payloads are rejected before a single bit is written.
- **Deterministic detection**: chi-square over the LSB histogram, with a
  fixed threshold and a *normalized* statistic (`ratio = chi2 / pairs`):
  the synthetic clean cover scores **96.76**, the fully embedded one
  **1.05** — a wide, non-borderline gap (threshold 4.0).

## Architecture

| Module | Responsibility |
|---|---|
| `bmp.py` | `Image`, `write_bmp`/`read_bmp`, `gradient_image`/`random_image` fixtures |
| `stego.py` | `capacity_bits/bytes`, `embed`, `extract`, HMAC header, file helpers |
| `detect.py` | LSB histogram, chi-square statistic, `detect`, `format_report` |
| `cli.py` | `embed`, `extract`, `analyze`, `capacity` commands |

Hidden stream layout (bits written MSB-first, low plane first, raster
order): `magic "OSSS" (4B) ‖ version/planes (1B) ‖ length (4B BE) ‖
HMAC-SHA256[:16] (16B) ‖ payload`.

## Runnable example

```console
$ python -c "
from bmp import gradient_image, save_bmp
save_bmp('cover.bmp', gradient_image(64, 64))
open('secret.txt','wb').write(b'Meet at the fountain at midnight')
"
$ python cli.py capacity --in cover.bmp
cover.bmp: 64x64, planes=0 -> 1511 bytes
$ python cli.py embed --in cover.bmp --out stego.bmp --key hunter2 --file secret.txt
embedded 32 bytes -> stego.bmp (capacity 1511 bytes, planes=0)
$ python cli.py extract --in stego.bmp --key hunter2 --out recovered.txt
extracted 32 bytes -> recovered.txt (HMAC ok)
$ python -c "print('recovered:', open('recovered.txt','rb').read().decode())"
recovered: Meet at the fountain at midnight
$ python cli.py analyze --in cover.bmp
cover.bmp: 64x64
chi2=12288.000  ratio=96.7559  threshold=4.00  pairs=127  odd/even=0.0000
verdict: clean (no signal)
$ python cli.py extract --in stego.bmp --key wrongkey --out x.bin
error: HMAC verification failed: wrong key or tampered image
```

Detection against a fully embedded image (payload = full 1511-byte
capacity of random bytes):

```console
$ python cli.py embed --in cover.bmp --out full.bmp --key hunter2 --file big.bin
embedded 1511 bytes -> full.bmp (capacity 1511 bytes, planes=0)
$ python cli.py analyze --in full.bmp
full.bmp: 64x64
chi2=133.101  ratio=1.0480  threshold=4.00  pairs=127  odd/even=0.9730
verdict: SUSPICIOUS (likely embedded)
```

## Honest limitations (documented, not hidden)

- **Partial embeddings evade chi-square.** The 32-byte example above is
  *not* flagged: it rewrites the LSB of only `(25+32)×8 = 456` of 12288
  bytes, leaving the histogram biased. Detection needs the payload to
  randomize a large share of the image — tests embed full capacity.
- **Uniform-LSB covers false-positive.** A cover that already has random
  LSBs (noise, encrypted content) scores ~1.0 and flags as suspicious —
  it is statistically indistinguishable from an embedded image.
- Planes 1–3 add capacity but are *more* visible (they perturb more
  bits); plane 0 is the subtlest. The detector inspects plane 0 only.

## Tests

```console
python -m unittest discover -s tests -v
```

71 tests: BMP write→read byte-exact roundtrips (including bottom-up BGR
row order and 4-byte padding), strict rejection of malformed BMPs,
embed→extract byte-exact at all four plane settings, exact capacity math
(1511/3047/4583/6119 bytes at planes 0–3 for the 64×64 cover), oversized
payloads rejected, wrong key and single-bit tampering fail cleanly via
HMAC, chi-square passes the clean gradient (ratio 96.76) and flags the
fully embedded image (ratio 1.05) with exact hand-computed statistic
values, and CLI end-to-end flows with correct exit codes (0/1/2).
