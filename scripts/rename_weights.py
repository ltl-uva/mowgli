import argparse
from collections import OrderedDict
import torch


def rename_weights(load_path: str, save_path: str, src: list, trg: list) -> None:
    """
    Renames weights of universal model to appropriate naming for a modular model.
    Example:
    `module.encoder.layer_norm.bias` -> `module.bg.encoder.layer_norm.bias` (where bg is a source language)
    """
    ckpt = torch.load(load_path)

    # Rename keys (remove `module`) when DDP model is loaded on single device
    if all([m.split(".")[0] == "module" for m in ckpt.keys()]):
        ckpt = {".".join(k.split(".")[1:]): v for k, v in ckpt.items()}

    model_state = OrderedDict()

    for k in ckpt["model_state"].keys():
        if ".encoder." in k or "src_embed" in k:
            for s in src:
                new_k = k.split(".")
                new_k.insert(2, s)
                model_state[".".join(new_k)] = ckpt["model_state"][k]

        elif ".decoder" in k or "trg_embed" in k:
            for t in trg:
                new_k = k.split(".")
                new_k.insert(2, t)
                model_state[".".join(new_k)] = ckpt["model_state"][k]

        else:
            print("Missing key: ", k)


    ckpt["model_state"] = model_state

    torch.save(ckpt, save_path)


if __name__ == "__main__":
    # Example usage: python rename_weights.py --load_path /path/to/ckpt.ckpt --save_path /path/to/new_ckpt.ckpt --src bg pl --trg en
    ap = argparse.ArgumentParser("Rename weights: universal to modular")
    ap.add_argument("--load_path", type=str, help="path to checkpoint")
    ap.add_argument("--save_path", type=str, help="path to newly created checkpoint")
    ap.add_argument("--src", nargs="+", help="source language(s)")
    ap.add_argument("--trg", nargs="+", help="target language(s)")

    args = ap.parse_args()

    assert args.load_path and args.save_path and args.src and args.trg
    assert args.load_path != args.save_path

    rename_weights(load_path=args.load_path, save_path=args.save_path, src=args.src, trg=args.trg)
