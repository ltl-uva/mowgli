"""
This script makes a slurm job file based on command output specifications, saves it to disk and launches the job.

Usage:
    python make_and_launch_slurm_job.py --mode {test/train} --config {path_to_config}

e.g.:
    python make_and_launch_slurm_job.py --mode test --config configs/test.yaml --gpu_type volta data.src:[de,es] data.trg:[en]
"""

import argparse
import os
import sys


def make_and_launch_slurm_job(args: argparse.Namespace, override_args: list):
    """
    Creates a slurm .job file from specified arguments. Saves and launches job after approval from user.

    :param args: mode (train or test), config path and SLURM related arguments
    :param override_args: mowgli arguments that override config
    """
    if not args.mem:    args.mem = 32  * args.n_gpu
    if not args.n_cpu:  args.n_cpu = 8 * args.n_gpu

    exp =   "\nexport OMP_NUM_THREADS=8 \n\n" if args.n_gpu > 1 else "\n"
    cmd =   f"python -u -m torch.distributed.launch --nproc_per_node={args.n_gpu}" if args.n_gpu > 1 else "python"
    log =   args.config.split('/')[-1].split('.')[0] if not args.slurm_log else args.slurm_log
    pip =   "pip install ${HOME}/code/mowgli/. \n\n" if not args.skip_install else "\n"

    job =   "#!/bin/sh \n\n" + \
            f"#SBATCH --mem={args.mem}G \n" + \
            f"#SBATCH --cpus-per-task={args.n_cpu} \n" + \
            f"#SBATCH --time={args.hours}:00:00 \n" + \
            f"#SBATCH --partition=gpu \n" + \
            f"#SBATCH --gres=gpu:{args.gpu_type}:{args.n_gpu} \n" + \
            f"#SBATCH --output=slurm/output/{log}.job \n" + \
            f"{exp}" + \
             "source activate mowgli \n" + \
            f"{pip}" + \
            f"{cmd} mowgli train {'${HOME}'}/code/mowgli/{args.config} {' '.join(override_args)} \n"

    print(job)

    proceed = None
    while proceed not in ["y", "n"]:
        proceed = input("Save and launch this job? y/n ")

    if proceed == "n":
        sys.exit("Change command line arguments.")

    slurm_job_fn = "slurm/" + args.config.split('/')[-1].split('.')[0] + ".job"
    print(f"Saving job at {slurm_job_fn}")
    with open(slurm_job_fn, "w") as f:
        f.write(job)

    os.system(f"sbatch {slurm_job_fn}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, choices=["train", "test"])
    ap.add_argument("--slurm_log", type=str)
    ap.add_argument("--skip_install", action="store_true")

    ap.add_argument("--config", type=str)
    ap.add_argument("--hours", type=int, default=240)
    ap.add_argument("--n_gpu", type=int, default=1)
    ap.add_argument("--n_cpu", type=int)
    ap.add_argument("--gpu_type", type=str, default="pascalxp")
    ap.add_argument("--mem", type=int)

    args, override_args = ap.parse_known_args()

    assert args.mode and args.config
    assert args.gpu_type in ["pascalxp", "volta"]

    make_and_launch_slurm_job(args=args, override_args=override_args)
