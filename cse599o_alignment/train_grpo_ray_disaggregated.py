"""
GRPO Skeleton: Minimal Asynchronous Training Loop
------------------------------------------------
Students should complete the TODO parts to:
 - implement rollout generation with reward computation using TransformerLM (Generator)
 - perform policy updates using GRPO algorithm (Learner)
 - synchronize model weights between Generator and Learner
"""

import argparse
import time
import numpy as np
import ray
from ray.experimental.collective import create_collective_group

from cse599o_alignment.train_util import make_keyword_inclusion_prompt, set_seed
from cse599o_alignment.train_grpo_ray_colocated import (
    N_GRPO_STEPS,
    Trajectory,
    Generator,
    Learner,
    split_keywords,
)


# ===================== Actors =====================


@ray.remote(num_gpus=1)
class GeneratorWorker(Generator):
    """Generator Ray actor."""

    def __init__(self, ckpt_file: str):
        super().__init__(ckpt_file=ckpt_file)


@ray.remote(num_gpus=1)
class LearnerWorker(Learner):
    """Learner Ray actor."""

    def __init__(self, ckpt_file: str):
        super().__init__(ckpt_file=ckpt_file)

    @ray.method(tensor_transport="nccl")
    def get_weights_rdt(self, **kwargs):
        return self.get_weights(**kwargs)


# ===================== Training loop =====================


def train_disaggregated(
    generator: "ray.actor.ActorHandle",
    learner: "ray.actor.ActorHandle",
    num_steps: int,
    keywords: list[str],
    prompts_per_batch: int,
    rdt: bool = False,
    profile: bool = False,
    verbose: bool = False,
) -> float:
    if num_steps < 1:
        return 0.0

    prompts: list[list[str]] = []
    for _ in range(num_steps):
        if not profile:
            kws_batch = np.random.choice(
                keywords, size=prompts_per_batch, replace=False
            )
        else:
            kws_batch = [keywords[0]] * prompts_per_batch
        prompts_batch = [make_keyword_inclusion_prompt([kw]) for kw in kws_batch]
        prompts.append(prompts_batch)

    trajs_ref: list[Trajectory] = generator.generate_trajectories.remote(
        prompts[0], verbose=verbose, profile=profile
    )
    transfer_ref = None
    for step in range(num_steps - 1):
        loss_ref = learner.update_policy.remote(
            trajs_ref, verbose=verbose, profile=profile
        )
        # Generate new trajectories for next step
        # Second rollout does not wait
        # Third rollout waits for first update and transfer
        trajs_ref = generator.generate_trajectories.remote(
            prompts[step + 1],
            verbose=verbose,
            profile=profile,
            transfer_ref=transfer_ref,
        )
        # Sync weights
        # Wait for update to complete
        if rdt:
            weights_ref = learner.get_weights_rdt.remote(loss_ref=loss_ref)
        else:
            weights_ref = learner.get_weights.remote(loss_ref=loss_ref)
        transfer_ref = generator.set_weights.remote(weights_ref)
    # Final step
    loss_ref = learner.update_policy.remote(trajs_ref, verbose=verbose, profile=profile)
    final_loss = ray.get(loss_ref)
    return final_loss


def run_training(
    keywords_file: str,
    train_val_kw_dir: str,
    ckpt_file: str,
    result_dir: str,
    num_steps: int = 10,
    num_workers: int = 2,
    prompts_per_batch: int = 32,
    num_val_prompts: int = 32,
    rdt: bool = False,
    verbose: bool = False,
    profile: bool = False,
) -> None:
    """
    Run colocated GRPO training with text generation.

    Args:
        keywords_file (str): Path to keywords file.
        train_val_kw_dir (str): Directory for train/val keyword split.
        ckpt_file (str): Path to model checkpoint.
        result_dir (str): Directory to save results.
        num_steps (int): Number of training steps to perform.
        num_workers (int): Number of colocated workers to use.
        prompts_per_batch (int): Number of prompts per training batch.
        num_val_prompts (int): Number of validation prompts.
        rdt (bool): Whether to use Ray Direct Transport (RDT).
        verbose (bool): Whether to enable verbose logging.
        profile (bool): Whether to enable profiling.
    """
    if num_workers != 2:
        raise ValueError("Only supports 2 workers for disaggregated training")
    if prompts_per_batch <= 0 or num_val_prompts <= 0:
        raise ValueError("prompts_per_batch and num_val_prompts must be positive")

    keywords_train, _ = split_keywords(
        train_val_kw_dir,
        keywords_file,
        prompts_per_batch,
        num_val_prompts,
    )

    # Create workers
    generator = GeneratorWorker.remote(ckpt_file=ckpt_file)
    learner = LearnerWorker.remote(ckpt_file=ckpt_file)
    if rdt:
        create_collective_group([generator, learner], backend="nccl")

    set_seed()
    num_warmup_steps = 2 if profile else 0
    train_args = dict(
        generator=generator,
        learner=learner,
        keywords=keywords_train,
        prompts_per_batch=prompts_per_batch,
        rdt=rdt,
        profile=profile,
        verbose=verbose,
    )
    warmup_loss = train_disaggregated(
        num_steps=num_warmup_steps,
        **train_args,
    )
    start_time = time.perf_counter()
    loss = train_disaggregated(
        num_steps=num_steps,
        **train_args,
    )
    end_time = time.perf_counter()
    elapse = end_time - start_time
    print(
        f"Warmup loss: {warmup_loss}, Final loss: {loss}, "
        f"Training time: {elapse:.2f} seconds"
    )
    # Get final statistics
    ray.get(learner.export_learner_stats.remote(result_dir))


def run_once(
    keywords_file: str,
    train_val_kw_dir: str,
    ckpt_file: str,
    result_dir: str,
    num_steps: int = 10,
    num_workers: int = 2,
    prompts_per_batch: int = 32,
    num_val_prompts: int = 32,
    rdt: bool = False,
    verbose: bool = False,
    profile: bool = False,
) -> None:
    """Entry point for training."""
    run_training(
        keywords_file=keywords_file,
        train_val_kw_dir=train_val_kw_dir,
        ckpt_file=ckpt_file,
        result_dir=result_dir,
        num_steps=num_steps,
        num_workers=num_workers,
        prompts_per_batch=prompts_per_batch,
        num_val_prompts=num_val_prompts,
        rdt=rdt,
        verbose=verbose,
        profile=profile,
    )


# ===================== Entry point =====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--keywords_file", type=str, required=True, help="Path to keywords file"
    )
    parser.add_argument(
        "--train_val_kw_split_dir",
        type=str,
        required=True,
        help="Directory for train/val keyword split",
    )
    parser.add_argument(
        "--ckpt_file", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--result_dir", type=str, required=True, help="Directory to save results"
    )

    parser.add_argument(
        "--steps", type=int, default=N_GRPO_STEPS, help="Number of training steps"
    )
    parser.add_argument(
        "--workers", type=int, default=2, help="Number of colocated workers"
    )
    parser.add_argument(
        "--prompts_per_batch", type=int, default=32, help="Number of prompts per batch"
    )
    parser.add_argument(
        "--num_val_prompts", type=int, default=32, help="Number of validation prompts"
    )
    parser.add_argument(
        "--rdt", action="store_true", help="Use Ray Direct Transport (RDT)"
    )
    parser.add_argument("--profile", action="store_true", help="Profiling flag")
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
        run_once(
            keywords_file=args.keywords_file,
            train_val_kw_dir=args.train_val_kw_split_dir,
            ckpt_file=args.ckpt_file,
            result_dir=args.result_dir,
            num_steps=args.steps,
            num_workers=args.workers,
            prompts_per_batch=args.prompts_per_batch,
            num_val_prompts=args.num_val_prompts,
            rdt=args.rdt,
            verbose=args.verbose,
            profile=args.profile,
        )
        log("Training completed.")
    finally:
        ray.shutdown()
        log("Ray shutdown.")
