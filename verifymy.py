#!/usr/bin/env python3
"""
H-ANS Codebase Provenance Verifier.

Scans all source files for hidden zero-width provenance markers
and verifies they match the registered dogtag.

Usage:
    python verifymy.py          -- Full scan with details
    python verifymy.py --quiet  -- Only show pass/fail

Exit code: 0 = verified, 1 = not found/mismatch
"""

import os
import sys
import glob
import hashlib
import argparse

EXPECTED_DOGTAG = "3732a5a3-05c9-43c7-be45-4746df4962b1"
EXPECTED_TOKEN = "ANS-HIJ-54CE6AD4"

ZW_CHARS = {
    "\u200b": "ZWSP",
    "\u200c": "ZWNJ",
    "\u200d": "ZWJ",
}

SCAN_PATTERNS = [
    "core/*.py",
    "fields/*.py",
    "paths/*.py",
    "collapse/*.py",
    "topology/*.py",
    "builds/**/*.py",
]


def decode_zw(text):
    """Extract string encoded in zero-width characters (16-bit encoding)."""
    bits = ""
    for ch in text:
        if ch == "\u200b":
            bits += "0"
        elif ch == "\u200c":
            bits += "1"
    # Convert 16-bit chunks to characters (matching encoder)
    chars = ""
    for i in range(0, len(bits), 16):
        word = bits[i : i + 16]
        if len(word) == 16:
            chars += chr(int(word, 2))
    return chars


def count_zw(text):
    """Count zero-width character locations."""
    count = 0
    for ch in text:
        if ch in ZW_CHARS:
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))

    # Gather files
    files = []
    for pat in SCAN_PATTERNS:
        files.extend(glob.glob(os.path.join(root, pat), recursive=True))
    files = sorted(set(f for f in files if os.path.isfile(f)))

    if not args.quiet:
        print("=" * 65)
        print("  H-ANS Provenance Verifier")
        print("=" * 65)
        print(f"  Dogtag: {EXPECTED_DOGTAG}")
        print(f"  Token:  {EXPECTED_TOKEN}")
        print(f"  Files:  {len(files)}")
        print("-" * 65)

    # Scan and decode
    all_decoded = ""
    found_files = []

    for fpath in files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            continue

        n = count_zw(content)
        if n > 0:
            d = decode_zw(content)
            all_decoded += d
            rel = os.path.relpath(fpath, root)
            found_files.append((rel, n))
            if not args.quiet:
                print(f"  [{n:4d} zw chars] {rel}")

    # Verify
    verified = EXPECTED_DOGTAG in all_decoded or EXPECTED_TOKEN in all_decoded

    if not args.quiet:
        print("-" * 65)

    if verified and found_files:
        print(f"  PASS - Provenance verified.")
        print(f"  Codebase created by owner of dogtag:")
        print(f"    {EXPECTED_DOGTAG}")
        print(f"    {EXPECTED_TOKEN}")
        print(f"  Found in {len(found_files)} files with dogtag watermark.")
        print("=" * 65)
        return 0
    elif found_files:
        print(f"  WARN - Watermarks found but don't match dogtag.")
        print(f"  Code may have been modified or regenerated.")
        print(f"  Decoded prefix: {ascii(all_decoded[:80])}")
        print("=" * 65)
        return 1
    else:
        print(f"  FAIL - No provenance watermarks found anywhere.")
        print(f"  Run: python tools/embed_dogtag.py")
        print(f"  See: PROVENANCE.md")
        print("=" * 65)
        return 1


if __name__ == "__main__":
    sys.exit(main())
