#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp


def read_metadata(ckpt):
    from torch.distributed.checkpoint import FileSystemReader
    reader = FileSystemReader(str(ckpt))
    return reader.read_metadata().state_dict_metadata


def shape_of(md):
    if hasattr(md, "size"):
        return tuple(md.size)
    if hasattr(md, "shape"):
        return tuple(md.shape)
    chunks = getattr(md, "chunks", None)
    if chunks:
        c = chunks[0]
        if hasattr(c, "sizes"):
            return tuple(c.sizes)
    raise RuntimeError(f"Cannot infer shape from metadata: {md}")


def dtype_of(md):
    props = getattr(md, "properties", None)
    if props is not None and hasattr(props, "dtype"):
        return props.dtype
    return torch.float32


def load_one(ckpt, key, md):
    state = {key: torch.empty(shape_of(md), dtype=dtype_of(md), device="cpu")}
    try:
        dcp.load(state, checkpoint_id=str(ckpt))
    except TypeError:
        from torch.distributed.checkpoint import FileSystemReader
        reader = FileSystemReader(str(ckpt))
        dcp.load_state_dict(state, storage_reader=reader, no_dist=True)
    return state[key].float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    args = ap.parse_args()

    md = read_metadata(args.ckpt)

    patch_keys = sorted([k for k in md if ".feed_forward.patch_feature_router." in k])
    hidden_keys = sorted([
        k for k in md
        if re.match(r"^model\.global_transformer\.layers\.\d+\.feed_forward\.router\.weight$", k)
    ])

    print(f"ckpt: {args.ckpt}")
    print(f"patch_feature_router keys: {len(patch_keys)}")
    print(f"hidden router weight keys: {len(hidden_keys)}")

    if patch_keys:
        print("\n[not hidden-only] Found patch_feature_router keys. First few:")
        for k in patch_keys[:10]:
            print(" ", k, shape_of(md[k]))
    else:
        print("\n[hidden-only/static-grid-N/A]")
        print("No patch_feature_router found.")
        print("This checkpoint cannot be compared with [log1p(length), entropy] static feature grid.")
        print("Use data-driven routing trace on the same evaluation samples instead.")

    if hidden_keys:
        print("\nHidden router weight diagnostics:")
        print("layer  shape          row_norm_mean  row_norm_std  weight_std")
        for k in hidden_keys:
            layer = int(k.split(".")[3])
            w = load_one(args.ckpt, k, md[k])
            row_norm = torch.linalg.vector_norm(w, dim=1)
            print(
                f"{layer:>5}  {str(tuple(w.shape)):<13} "
                f"{row_norm.mean().item():.4e}   "
                f"{row_norm.std(unbiased=False).item():.4e}   "
                f"{w.std(unbiased=False).item():.4e}"
            )


if __name__ == "__main__":
    main()
