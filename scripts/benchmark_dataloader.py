"""
Used to find an efficient number for `num_workers` and value for `pin_memory`.
This number is a function of the system (#CPUs, #GPUs, OMP_NUM_THREADS) and the batch size.

Usage [cpu or single GPU]:
  python scripts/benchmark_dataloader.py configs/config.yaml

Usage [multi-GPU]:
  python -u -m torch.distributed.launch --nproc_per_node={#GPUs} scripts/benchmark_dataloader.py configs/config.yaml
"""

import argparse
import time
import os
import torch
from mowgli.helpers import load_config, merge_iterators
from mowgli.data import build_datasets, build_iterator


def benchmark_dataloader(cfg: dict, rank: int):
    """
    Benchmarks dataloader for different combinations of `num_worker` and `pin_memory`.
    """

    print("Number of CPUs: ", os.cpu_count(), "Num_threads: ",torch.get_num_threads())
    device = torch.device("cuda" if cfg["training"]["use_cuda"] else "cpu", index=rank)

    distributed=False
    # initialize torch.distributed if using multiple GPUs
    if cfg["training"]["use_cuda"] and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print("initialize DistributedDataParallel...")
        distributed=True
        torch.distributed.init_process_group(backend='nccl')

    # Load datasets, vocabulary, and vocabulary keys
    train_data, valid_data, test_data, vocab, src_key, trg_key = build_datasets(cfg["data"], distributed=distributed)

    for num_workers in range(0, 32):
        for pin_memory in [True, False]:
            train_iter = merge_iterators(
                [
                    build_iterator(
                        dataset             = data,
                        dataset_type        = data.type,
                        batch_size          = cfg["training"]["batch_size"],
                        batch_type          = cfg["training"]["batch_type"],
                        train               = True,
                        pad_idx             = 0,
                        shuffle             = True,
                        sample_temperature  = cfg["training"].get("sample_temperature", 1),
                        distributed         = distributed,
                        num_workers         = num_workers,
                        pin_memory          = pin_memory,
                    )
                    for data in train_data
                ],
                batch_multiplier    = cfg["training"]["batch_multiplier"],
                datasets            = train_data,
            )

            start = time.time()
            for epoch in range(1,3):
                for idx, batch in enumerate(train_iter):
                    batch.to_device(device)
            end = time.time()
            print("(Worker {}) Finish after: {} (num_workers={}, pin_memory={})".format(
                rank, end-start, num_workers, pin_memory)
            )


if __name__ == "__main__":
    ap = argparse.ArgumentParser("Mowgli NMT")
    ap.add_argument("config_path", type=str, help="path to YAML config file")
    ap.add_argument("--local_rank", default=0, type=int, help="Local rank (for distributed training).")
    args, override_args = ap.parse_known_args()

    benchmark_dataloader(cfg=load_config(args.config_path), rank=args.local_rank)
