#!/usr/bin/env python3
"""bin-2-q8 — Convert binary data to Quate (Q8) pronounceable encoding.

Each nibble maps to a consonant-vowel syllable:
  Consonant (high 2 bits): 00=' 01=J 10=K 11=V
  Vowel     (low 2 bits):  00=O 01=U 10=E 11=I

Two syllables per byte, 8 words per line.

Usage:
  bin-2-q8.py <file>           # encode file contents
  bin-2-q8.py -b <binary-str>  # encode literal binary string (e.g. "10010010")
  echo "hello" | bin-2-q8.py   # encode stdin
"""

import sys
import argparse

CONSONANTS = ("'", "J", "K", "V")
VOWELS = ("O", "U", "E", "I")


def nibble_to_syllable(nibble: int) -> str:
    """Convert a 4-bit value to a consonant-vowel syllable."""
    consonant = CONSONANTS[(nibble >> 2) & 0x3]
    vowel = VOWELS[nibble & 0x3]
    return consonant + vowel


def byte_to_word(b: int) -> str:
    """Convert a byte to a two-syllable Q8 word."""
    high = (b >> 4) & 0xF
    low = b & 0xF
    return nibble_to_syllable(high) + nibble_to_syllable(low)


def bits_to_bytes_with_checksum(bits: str) -> bytes:
    """Convert a bit string to bytes, XOR-padding any trailing sub-byte bits."""
    # Strip anything that isn't 0 or 1
    bits = "".join(c for c in bits if c in "01")
    if not bits:
        return b""

    remainder = len(bits) % 8
    if remainder:
        # XOR all complete bytes to derive checksum bits
        full_bytes = len(bits) // 8
        xor_acc = 0
        for i in range(full_bytes):
            byte_bits = bits[i * 8 : (i + 1) * 8]
            xor_acc ^= int(byte_bits, 2)

        # Also fold in the partial bits so far
        partial = bits[full_bytes * 8 :]
        partial_val = int(partial, 2)
        xor_acc ^= partial_val

        # Take the needed number of low bits from the checksum
        pad_needed = 8 - remainder
        pad_bits = format(xor_acc & ((1 << pad_needed) - 1), f"0{pad_needed}b")
        bits += pad_bits

    return int(bits, 2).to_bytes(len(bits) // 8, "big")


def encode_q8(data: bytes) -> str:
    """Encode bytes into Q8 text: 8 words per line, space-separated."""
    words = [byte_to_word(b) for b in data]
    lines = []
    for i in range(0, len(words), 8):
        lines.append(" ".join(words[i : i + 8]))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Convert binary data to Quate (Q8) encoding."
    )
    parser.add_argument("file", nargs="?", help="File to encode (raw bytes)")
    parser.add_argument(
        "-b", "--binary", help='Literal binary string (e.g. "10010010")'
    )
    args = parser.parse_args()

    if args.binary:
        data = bits_to_bytes_with_checksum(args.binary)
    elif args.file:
        with open(args.file, "rb") as f:
            data = f.read()
    elif not sys.stdin.isatty():
        data = sys.stdin.buffer.read()
    else:
        parser.print_help()
        sys.exit(1)

    if data:
        print(encode_q8(data))


if __name__ == "__main__":
    main()
