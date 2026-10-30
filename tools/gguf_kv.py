#!/usr/bin/env python3
"""Print the metadata keys of a GGUF file without numpy: key, type and value (arrays as their
length and first few items). Tokenizer arrays are summarised.
usage: gguf_kv.py file.gguf [key-substring]
"""
import struct
import sys

_SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
_STRING, _ARRAY = 8, 9


def _read(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))


def _read_str(f):
    (n,) = _read(f, "Q")
    return f.read(n).decode("utf-8", "replace")


def _read_value(f, vtype):
    if vtype in _SCALAR:
        return _read(f, _SCALAR[vtype])[0]
    if vtype == _STRING:
        return _read_str(f)
    if vtype == _ARRAY:
        etype, n = _read(f, "IQ")
        items = [_read_value(f, etype) for _ in range(n)]
        return items
    raise ValueError(f"unknown GGUF value type {vtype}")


def main():
    path = sys.argv[1]
    filt = sys.argv[2] if len(sys.argv) > 2 else ""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            sys.exit(f"{path}: not a GGUF file")
        _version, _n_tensors, n_kv = _read(f, "IQQ")
        for _ in range(n_kv):
            key = _read_str(f)
            (vtype,) = _read(f, "I")
            val = _read_value(f, vtype)
            if filt not in key:
                continue
            if isinstance(val, list):
                head = val[:8] if not key.startswith("tokenizer.") else val[:3]
                print(f"{key}\tarray[{len(val)}]\t{head}{' ...' if len(val) > len(head) else ''}")
            else:
                print(f"{key}\t{val}")


if __name__ == "__main__":
    main()
