import argparse
import json
import time
import numpy as np
import ray
from ray.experimental.collective import create_collective_group

from cse599o_alignment.train_grpo_ray_disaggregated import (
    GeneratorWorker,
    LearnerWorker,
)


def rdt_step(
    generator: "ray.actor.ActorHandle",
    learner: "ray.actor.ActorHandle",
    rdt: bool,
    verbose: bool = False,
) -> float:
    start_time = time.perf_counter()
    if rdt:
        weights_ref = learner.get_weights_rdt.remote()
    else:
        weights_ref = learner.get_weights.remote()
    ray.get(generator.set_weights.remote(weights_ref))
    end_time = time.perf_counter()
    elapse = end_time - start_time
    if verbose:
        print(f"(RDT={rdt}) Time: {elapse:.4f} seconds")
    return elapse


def benchmark_rdt(
    ckpt_file: str,
    result_dir: str,
    num_steps: int,
    num_warmup_steps: int,
    verbose: bool = False,
) -> None:
    # Create workers
    generator = GeneratorWorker.remote(ckpt_file=ckpt_file)
    learner = LearnerWorker.remote(ckpt_file=ckpt_file)
    create_collective_group([generator, learner], backend="nccl")

    for _ in range(num_warmup_steps):
        rdt_step(
            generator=generator,
            learner=learner,
            rdt=True,
            verbose=verbose,
        )
    if verbose:
        print("(RDT) Warmup completed. Starting benchmark...")
    rdt_times = []
    for _ in range(num_steps):
        rdt_times.append(
            rdt_step(
                generator=generator,
                learner=learner,
                rdt=True,
                verbose=verbose,
            )
        )

    for _ in range(num_warmup_steps):
        rdt_step(
            generator=generator,
            learner=learner,
            rdt=False,
            verbose=verbose,
        )
    if verbose:
        print("(No RDT) Warmup completed. Starting benchmark...")
    no_rdt_times = []
    for _ in range(num_steps):
        no_rdt_times.append(
            rdt_step(
                generator=generator,
                learner=learner,
                rdt=False,
                verbose=verbose,
            )
        )

    avg_rdt_time = np.mean(rdt_times)
    std_rdt_time = np.std(rdt_times)
    avg_no_rdt_time = np.mean(no_rdt_times)
    std_no_rdt_time = np.std(no_rdt_times)
    speedup = avg_no_rdt_time / avg_rdt_time
    if verbose:
        print(f"RDT: {avg_rdt_time:.4f} ± {std_rdt_time:.4f} seconds per transfer")
        print(
            f"No RDT: {avg_no_rdt_time:.4f} ± {std_no_rdt_time:.4f} seconds per transfer"
        )
        print(f"Speedup: {speedup:.2f}x")

    output_file = f"{result_dir}/rdt.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "rdt": {
                    "avg_time": avg_rdt_time,
                    "std_time": std_rdt_time,
                },
                "no_rdt": {
                    "avg_time": avg_no_rdt_time,
                    "std_time": std_no_rdt_time,
                },
                "speedup": speedup,
            },
            f,
            indent=4,
        )


# ===================== Entry point =====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_file", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--result_dir", type=str, required=True, help="Directory to save results"
    )

    parser.add_argument(
        "--steps", type=int, default=10, help="Number of training steps"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=3, help="Number of warmup steps"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    def log(msg: str) -> None:
        if args.verbose:
            print(msg)

    log(f"Args: {args}.")

    ray.init(
        runtime_env={
            "excludes": [
                ".git/**",  # git metadata and objects
                ".venv/**",  # virtual environment
                "tests/fixtures/**",  # test fixtures (large model files)
                "*.nsys-rep",  # profiling files
                # "*.pt",
                # "*.pth",
                # "*.safetensors",  # model weight files
                "*.tar",
                "*.zip",
                "*.gz",  # archives
                "__pycache__/**",  # Python cache
                "*.egg-info/**",  # package info
            ]
        },
        _temp_dir="/homes/iws/yhruan22/raytmp",
        ignore_reinit_error=True,
    )
    log("Ray initialized.")

    try:
        benchmark_rdt(
            ckpt_file=args.ckpt_file,
            result_dir=args.result_dir,
            num_steps=args.steps,
            num_warmup_steps=args.warmup_steps,
            verbose=args.verbose,
        )
        log("Training completed.")
    finally:
        ray.shutdown()
        log("Ray shutdown.")
