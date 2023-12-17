import json
import os
import sys
import copy
import torch
from safetensors.torch import load_file, save_file

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, help="Path to input pth model")
parser.add_argument("--output", type=str, help="Path to output safetensors model")
args = parser.parse_args()


def rename_key(rename, name):
    for k, v in rename.items():
        if k in name:
            name = name.replace(k, v)
    return name


def convert_file(pt_filename: str, sf_filename: str, transpose_names=[]):
    loaded = torch.load(pt_filename, map_location="cpu")
    if "state_dict" in loaded:
        loaded = loaded["state_dict"]

    # huggingface permutes WQ and WK, this function reverses it
    def permute_reverse(w, n_heads=16, dim1=2048, dim2=2048):
        return w.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)

    # For tensors to be contiguous
    for k, v in loaded.items():
        for transpose_name in transpose_names:
            if transpose_name in k:
                print("transpose", k),
                loaded[k] = permute_reverse(v)

    # fp16
    # loaded = {k: v.clone().half().contiguous() for k, v in loaded.items()}
    # fp32
    loaded = {k: v.clone().contiguous() for k, v in loaded.items()}

    for k, v in loaded.items():
        print(f"{k}\t{v.shape}\t{v.dtype}")

    save_file(loaded, sf_filename, metadata={"format": "pt"})
    reloaded = load_file(sf_filename)
    for k in loaded:
        pt_tensor = loaded[k]
        sf_tensor = reloaded[k]
        if not torch.equal(pt_tensor, sf_tensor):
            raise RuntimeError(f"The output tensors do not match for key {k}")


if __name__ == "__main__":
    convert_file(args.input, args.output, transpose_names=["q_proj", "k_proj"])
    print(f"Saved to {args.output}")
