"""Minimal, dependency-free reader for .safetensors files (used when the `safetensors` package is absent).

Format: 8-byte little-endian header length, JSON header {name: {dtype, shape, data_offsets}}, raw bytes.
Only the dtypes HVI-CIDNet checkpoints use are supported; anything else raises.
"""
import json
import struct

import numpy as np
import torch

_DT = {"F32": np.float32, "F16": np.float16, "I64": np.int64, "I32": np.int32}


def load_file(path):
    try:
        from safetensors.torch import load_file as _lf
        return _lf(path)
    except ImportError:
        pass
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        f.seek(0)
        blob = f.read()
    out = {}
    for k, v in header.items():
        if k == "__metadata__":
            continue
        if v["dtype"] not in _DT:
            raise ValueError(f"unsupported dtype {v['dtype']} for {k}")
        a, b = v["data_offsets"]
        arr = np.frombuffer(blob[base + a:base + b], dtype=_DT[v["dtype"]]).reshape(v["shape"])
        out[k] = torch.from_numpy(arr.copy())
    return out
