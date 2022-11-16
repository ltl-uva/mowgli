import argparse
import shutil

from mowgli.training import train
from mowgli.prediction import test
from mowgli.helpers import load_config, make_model_dir


def main():
    ap = argparse.ArgumentParser("Mowgli NMT")
    ap.add_argument("mode", choices=["train", "test"], help="train or test a model")
    ap.add_argument("config_path", type=str, help="path to YAML config file")
    ap.add_argument("--ckpt", type=str, help="checkpoint for prediction")
    ap.add_argument("--output_path", type=str, help="path for saving translation output")
    ap.add_argument("--save_attention", action="store_true", help="save attention visualizations")
    ap.add_argument("--local_rank", default=0, type=int, help="Local rank (for distributed training).")
    args, override_args = ap.parse_known_args()

    cfg = load_config(args.config_path, override_args=override_args)

    if args.mode == "train":
        # create model directory
        model_dir = make_model_dir(
            cfg["training"]["model_dir"],
            overwrite=cfg["training"].get("overwrite", False),
            rank=args.local_rank
        )

        # store copy of original training config in model dir
        if args.local_rank == 0:
            shutil.copy2(args.config_path, cfg["training"]["model_dir"] + "/config.yaml")

        train(cfg, args.local_rank)

    elif args.mode == "test":
        test(cfg=cfg, ckpt=args.ckpt, output_path=args.output_path, save_attention=args.save_attention)

    else:
        raise ValueError("Unknown mode")


if __name__ == "__main__":
    main()
