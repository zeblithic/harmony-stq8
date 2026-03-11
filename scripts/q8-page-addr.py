#!/usr/bin/env python3
"""q8-page-addr — Q8 page address encoder with multi-format display.

Computes SHA-256 and SHA-224 hashes of input data, then derives four
32-bit page addresses (MSB/LSB x SHA-256/SHA-224) with 2-bit type tags
and 2-bit XOR checksums.

Output formats:
  Hex:   raw hexadecimal
  Q8:    Q8-FLAT phonetic (consonant+vowel syllable pairs)
  Box:   Q8-BOX split display (consonants / vowels on separate lines)
  BIP39: BIP-0039 mnemonic words (11-bit word indices into 2048-word list)

Page address layout: [2 mode bits][28 data bits][2 XOR checksum bits]

    Mode '00': MSB SHA-256 (first 28 bits)
    Mode '01': LSB SHA-256 (last 28 bits)
    Mode '10': MSB SHA-224 (first 28 bits)
    Mode '11': LSB SHA-224 (last 28 bits)

Usage:
    q8-page-addr.py              # 4KB random noise demo
    q8-page-addr.py <file>       # hash file contents
    echo data | q8-page-addr.py  # hash stdin
"""

import sys
import os
import hashlib

from mnemonic import Mnemonic

_BIP39 = Mnemonic("english")

# Q8-BOX character encoding (visual grid)
BOX_CONSONANTS = ("A", ">", "<", "V")  # 00=glottal 01=J 10=K 11=V
BOX_VOWELS     = ("O", "=", "X", "I")  # 00=O 01=U 10=E 11=I

# Q8-FLAT character encoding (phonetic)
FLAT_CONSONANTS = ("'", "J", "K", "V")  # 00=glottal 01=J 10=K 11=V
FLAT_VOWELS     = ("O", "U", "E", "I")  # 00=O 01=U 10=E 11=I


def nibble_box_consonant(nibble: int) -> str:
    return BOX_CONSONANTS[(nibble >> 2) & 0x3]


def nibble_box_vowel(nibble: int) -> str:
    return BOX_VOWELS[nibble & 0x3]


def nibble_to_flat_syllable(nibble: int) -> str:
    return FLAT_CONSONANTS[(nibble >> 2) & 0x3] + FLAT_VOWELS[nibble & 0x3]


def byte_to_flat_word(b: int) -> str:
    return nibble_to_flat_syllable((b >> 4) & 0xF) + nibble_to_flat_syllable(b & 0xF)


def format_hex(data: bytes) -> str:
    return " ".join(f"{b:02x}" for b in data)


def format_flat(data: bytes) -> str:
    words = [byte_to_flat_word(b) for b in data]
    lines = [" ".join(words[i:i + 8]) for i in range(0, len(words), 8)]
    return "\n".join(lines)


def format_split_box(data: bytes, bytes_per_row: int = 8) -> str:
    """Split box: consonants on top, vowels on bottom, grouped by byte."""
    rows = []
    for row_start in range(0, len(data), bytes_per_row):
        row_bytes = data[row_start:row_start + bytes_per_row]
        cons = []
        vows = []
        for b in row_bytes:
            high = (b >> 4) & 0xF
            low = b & 0xF
            cons.append(nibble_box_consonant(high) + nibble_box_consonant(low))
            vows.append(nibble_box_vowel(high) + nibble_box_vowel(low))
        rows.append(" ".join(cons))
        rows.append(" ".join(vows))
        rows.append("")
    if rows and rows[-1] == "":
        rows.pop()
    return "\n".join(rows)


def format_bip39(data: bytes, words_per_line: int = 6) -> str:
    """BIP-0039 mnemonic for standard entropy sizes (16/20/24/28/32 bytes).

    Includes the BIP39 checksum (first entropy_bits/32 bits of SHA-256).
    """
    mnemonic = _BIP39.to_mnemonic(data)
    words = mnemonic.split()
    lines = [" ".join(words[i:i + words_per_line])
             for i in range(0, len(words), words_per_line)]
    return "\n".join(lines)


def format_bip39_checksummed(data: bytes, words_per_line: int = 3) -> str:
    """BIP39-style encoding for arbitrary-length data.

    Appends entropy_bits/32 checksum bits from SHA-256(data), then
    splits into 11-bit word indices. Works for any byte length where
    (entropy_bits + checksum_bits) is divisible by 11.
    """
    wordlist = _BIP39.wordlist
    entropy_bits = len(data) * 8
    cs_bits = entropy_bits // 32

    cs_hash = hashlib.sha256(data).digest()
    cs_byte = cs_hash[0]  # first byte of SHA-256(entropy)

    # Build bit string: entropy || checksum
    bits = int.from_bytes(data, "big")
    bits = (bits << cs_bits) | (cs_byte >> (8 - cs_bits))
    total_bits = entropy_bits + cs_bits
    n_words = total_bits // 11

    words = []
    for i in range(n_words):
        idx = (bits >> (total_bits - 11 * (i + 1))) & 0x7FF
        words.append(wordlist[idx])

    lines = [" ".join(words[i:i + words_per_line])
             for i in range(0, len(words), words_per_line)]
    return "\n".join(lines)


def make_page_address(hash_bytes: bytes, variant: int) -> bytes:
    """Build a 32-bit page address.

    Layout: [2 mode bits][28 data bits][2 XOR checksum bits] = 32 bits

    Mode as MSBs so the first consonant heard immediately identifies
    the hash variant (00=MSB-256, 01=LSB-256, 10=MSB-224, 11=LSB-224).

    XOR checksum: fold all 15 two-bit pairs of the 30-bit prefix
    (2 mode + 28 data) into a single 2-bit value.
    """
    hash_int = int.from_bytes(hash_bytes, "big")
    total_bits = len(hash_bytes) * 8

    if variant % 2 == 0:  # MSB — first 28 bits
        data_28 = (hash_int >> (total_bits - 28)) & 0x0FFFFFFF
    else:                  # LSB — last 28 bits
        data_28 = hash_int & 0x0FFFFFFF

    # Pack 30-bit prefix: mode(2) | data(28)
    val_30 = ((variant & 0x3) << 28) | data_28

    # XOR checksum: fold 15 two-bit pairs
    xor = 0
    for i in range(15):
        xor ^= (val_30 >> (i * 2)) & 0x3

    address = (val_30 << 2) | (xor & 0x3)
    return address.to_bytes(4, "big")


def main():
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        with open(filepath, "rb") as f:
            data = f.read()
        print(f"Input: {filepath} ({len(data)} bytes)\n")
    elif not sys.stdin.isatty():
        data = sys.stdin.buffer.read()
        if not data:
            data = os.urandom(4096)
            print(f"(Generated {len(data)} bytes of random noise)\n")
        else:
            print(f"Input: stdin ({len(data)} bytes)\n")
    else:
        data = os.urandom(4096)
        print(f"(Generated {len(data)} bytes of random noise)\n")

    sha256 = hashlib.sha256(data).digest()
    sha224 = hashlib.sha224(data).digest()

    # Full hashes
    print("══ SHA-256 (24 BIP39 words) ══")
    print(f"Hex:\n{format_hex(sha256)}")
    print(f"Flat:")
    print(format_flat(sha256))
    print(f"BIP39:")
    print(format_bip39(sha256))
    print("Box:")
    print(format_split_box(sha256))
    print()

    print("══ SHA-224 (21 BIP39 words) ══")
    print(f"Hex:\n{format_hex(sha224)}")
    print(f"Flat:")
    print(format_flat(sha224))
    print(f"BIP39:")
    print(format_bip39(sha224))
    print("Box:")
    print(format_split_box(sha224))
    print()

    # Page addresses
    variants = [
        (0, "MSB SHA-256", "00", sha256),
        (1, "LSB SHA-256", "01", sha256),
        (2, "MSB SHA-224", "10", sha224),
        (3, "LSB SHA-224", "11", sha224),
    ]


    print("══ Page Addresses (32-bit) ══")
    print("Layout: [2 mode][28 data][2 XOR checksum]\n")

    for variant_id, label, tag, hash_bytes in variants:
        addr = make_page_address(hash_bytes, variant_id)
        print(f"── {label} (mode={tag}) ──")
        print(f"Hex:   {format_hex(addr)}")
        print(f"Flat:  {' '.join(byte_to_flat_word(b) for b in addr)}")
        print(f"BIP39: {format_bip39_checksummed(addr)}")
        print("Box:")
        print(format_split_box(addr, bytes_per_row=4))
        print()


if __name__ == "__main__":
    main()
