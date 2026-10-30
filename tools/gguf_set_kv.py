#!/usr/bin/env python3
"""Copy a GGUF file with metadata keys added or replaced, without numpy. The tensor infos and
data are copied unchanged (the data section is re-aligned after the new header).
usage: gguf_set_kv.py in.gguf out.gguf key=u32:N [key=f32:X] [key=u32[]:a,b,c] ...

Unsloth's DeepSeek-V4.1 DSpark drafter (arch dflash) lacks the DSA indexer keys that ik reads
for every DeepSeek-V4 stage, although its stages are uncompressed and never use them:
  gguf_set_kv.py dspark-Q8_0.gguf dspark-ik.gguf dflash.attention.indexer.head_count=u32:32 \\
      dflash.attention.indexer.key_length=u32:128 dflash.attention.indexer.top_k=u32:512
"""
import shutil
import struct
import sys

_SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
_STRING, _ARRAY = 8, 9
_TYPES = {"u8": 0, "i8": 1, "u16": 2, "i16": 3, "u32": 4, "i32": 5, "f32": 6, "bool": 7, "u64": 10, "i64": 11, "f64": 12}


def _read(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))


def _read_str(f):
    (n,) = _read(f, "Q")
    return f.read(n)


def _skip_value(f, vtype):
    """Read a value and return its raw bytes (type tag not included)."""
    start = f.tell()
    if vtype in _SCALAR:
        f.read(struct.calcsize("<" + _SCALAR[vtype]))
    elif vtype == _STRING:
        _read_str(f)
    elif vtype == _ARRAY:
        etype, n = _read(f, "IQ")
        if etype in _SCALAR:
            f.read(n*struct.calcsize("<" + _SCALAR[etype]))
        else:
            for _ in range(n):
                _skip_value(f, etype)
    else:
        raise ValueError(f"unknown GGUF value type {vtype}")
    end = f.tell()
    f.seek(start)
    return f.read(end - start)


def _str(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def _encode(spec):
    tname, _, val = spec.partition(":")
    if tname.endswith("[]"):
        etype = _TYPES[tname[:-2]]
        items = [float(v) if etype in (6, 12) else int(v) for v in val.split(",")]
        return struct.pack("<IIQ", _ARRAY, etype, len(items)) + b"".join(struct.pack("<" + _SCALAR[etype], v) for v in items)
    if tname == "str":
        return struct.pack("<I", _STRING) + _str(val)
    vtype = _TYPES[tname]
    v = float(val) if vtype in (6, 12) else (val.lower() in ("1", "true") if vtype == 7 else int(val))
    return struct.pack("<I", vtype) + struct.pack("<" + _SCALAR[vtype], v)


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    src, dst = sys.argv[1], sys.argv[2]
    sets = {}
    for arg in sys.argv[3:]:
        key, _, spec = arg.partition("=")
        sets[key] = _encode(spec)

    with open(src, "rb") as f:
        if f.read(4) != b"GGUF":
            sys.exit(f"{src}: not a GGUF file")
        version, n_tensors, n_kv = _read(f, "IQQ")
        kvs = []
        alignment = 32
        for _ in range(n_kv):
            key = _read_str(f).decode()
            (vtype,) = _read(f, "I")
            raw = _skip_value(f, vtype)
            if key == "general.alignment":
                alignment = struct.unpack("<I", raw)[0]
            kvs.append((key, struct.pack("<I", vtype) + raw))
        tinfo_start = f.tell()
        for _ in range(n_tensors):
            _read_str(f)
            (n_dims,) = _read(f, "I")
            f.read(8*n_dims + 4 + 8)
        tinfo_end = f.tell()
        f.seek(tinfo_start)
        tinfo = f.read(tinfo_end - tinfo_start)
        data_start = (tinfo_end + alignment - 1)//alignment*alignment

        out_kvs = []
        for key, val in kvs:
            if key in sets:
                print(f"replaced {key}")
                val = sets.pop(key)
            out_kvs.append((key, val))
        for key, val in sets.items():
            print(f"added {key}")
            out_kvs.append((key, val))

        header = b"GGUF" + struct.pack("<IQQ", version, n_tensors, len(out_kvs))
        header += b"".join(_str(k) + v for k, v in out_kvs) + tinfo
        header += b"\0"*((-len(header)) % alignment)

        with open(dst, "wb") as g:
            g.write(header)
            f.seek(data_start)
            shutil.copyfileobj(f, g, 64 << 20)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
