#!/usr/bin/env python3
"""List the tensors in one or more GGUF files (name, type, shape, bytes) without numpy.

Sizes come from the gap between consecutive tensor offsets, so unknown quant
types are still measured correctly. Output is TSV on stdout:
    file  name  type_id  shape  bytes
"""
import os
import struct
import sys

# GGUF metadata value types
_SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
_STRING, _ARRAY = 8, 9


def _read(f, fmt):
    size = struct.calcsize(fmt)
    return struct.unpack("<" + fmt, f.read(size))


def _read_str(f):
    (n,) = _read(f, "Q")
    return f.read(n).decode("utf-8", "replace")


def _skip_value(f, vtype):
    if vtype in _SCALAR:
        f.seek(struct.calcsize(_SCALAR[vtype]), 1)
    elif vtype == _STRING:
        (n,) = _read(f, "Q")
        f.seek(n, 1)
    elif vtype == _ARRAY:
        (etype,) = _read(f, "I")
        (count,) = _read(f, "Q")
        if etype in _SCALAR:
            f.seek(struct.calcsize(_SCALAR[etype]) * count, 1)
        else:
            for _ in range(count):
                _skip_value(f, etype)
    else:
        raise ValueError(f"unknown GGUF value type {vtype}")


def _read_value(f, vtype):
    if vtype in _SCALAR:
        return _read(f, _SCALAR[vtype])[0]
    if vtype == _STRING:
        return _read_str(f)
    _skip_value(f, vtype)
    return None


def tensors(path):
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        (version,) = _read(f, "I")
        n_tensors, n_kv = _read(f, "QQ")
        alignment = 32
        meta = {}
        for _ in range(n_kv):
            key = _read_str(f)
            (vtype,) = _read(f, "I")
            if vtype == _ARRAY:
                _skip_value(f, vtype)
                continue
            val = _read_value(f, vtype)
            meta[key] = val
            if key == "general.alignment":
                alignment = val
        infos = []
        for _ in range(n_tensors):
            name = _read_str(f)
            (n_dims,) = _read(f, "I")
            dims = _read(f, "Q" * n_dims)
            (ttype,) = _read(f, "I")
            (offset,) = _read(f, "Q")
            infos.append([name, ttype, dims, offset])
        data_start = (f.tell() + alignment - 1) // alignment * alignment
        file_size = os.fstat(f.fileno()).st_size
    order = sorted(range(len(infos)), key=lambda i: infos[i][3])
    for k, i in enumerate(order):
        end = infos[order[k + 1]][3] if k + 1 < len(order) else file_size - data_start
        infos[i].append(end - infos[i][3])
    return meta, infos


def main():
    for path in sys.argv[1:]:
        meta, infos = tensors(path)
        if "--meta" in os.environ.get("GGUF_OPTS", ""):
            for k, v in meta.items():
                print(f"# {os.path.basename(path)}\t{k}\t{v}", file=sys.stderr)
        for name, ttype, dims, _offset, size in infos:
            shape = "x".join(str(d) for d in dims)
            print(f"{os.path.basename(path)}\t{name}\t{ttype}\t{shape}\t{size}")


if __name__ == "__main__":
    main()
