"""
GRPO Skeleton: Colocated Synchronous Training Loop (Simplified)
--------------------------------------------------------------
Students should complete the TODO parts to:
 - implement rollout generation with reward computation using TransformerLM
 - perform policy updates using GRPO algorithm
 - implement keyword inclusion reward function

This version combines Generator and Learner into a single actor for simplified
synchronous training without replay buffer, training directly on each trajectory.
"""

import argparse
import ray
import torch
import tiktoken
import time
from typing import Any
import numpy as np

from cse599o_basics.model import Transformer
from cse599o_basics.optimizer import AdamW, gradient_clipping
from cse599o_alignment.grpo import grpo_microbatch_train_step
from cse599o_alignment.train_util import (
    extract_keywords_from_prompt,
    generate_response,
    keyword_inclusion_reward,
    make_keyword_inclusion_prompt,
)


# ===================== Basic setup =====================


VOCAB_SIZE = tiktoken.get_encoding("gpt2").n_vocab
CONTEXT_LENGTH = 256
NUM_LAYERS = 4
D_MODEL = 512
NUM_HEADS = 16
D_FF = 1344
THETA = 10000
CHECKPOINT_PATH = "ckpt/model.pt"

LEARNING_RATE = 5e-4
ADAMW_ARGS = {
    "lr": LEARNING_RATE,
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "weight_decay": 0.01,
}
MAX_GRAD_NORM = 1.0

N_GRPO_STEPS = 100
G = 4  # group size (number of responses per prompt)
SAMPLING_TEMPERATURE = 0.8
SAMPLING_MAX_TOKENS = 60
ADVANTAGE_EPS = 1e-8
LOSS_TYPE = "grpo_clip"
USE_STD_NORMALIZATION = True
CLIPRANGE = 0.2


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===================== Data container =====================


class Trajectory:
    """Stores a single rollout trajectory for text generation"""

    def __init__(
        self,
        prompt: str,
        responses: torch.Tensor,  # (G, sequence_length)
        rewards: torch.Tensor,  # (G,)
        log_probs: torch.Tensor,  # (G, sequence_length)
        response_masks: torch.Tensor,  # (G, sequence_length)
    ):
        """
        Args:
            prompt: The input prompt string.
            responses: Tensor of shape (G, sequence_length) for generated response
                tokens.
            rewards: Tensor of shape (G,) with rewards for each response.
            log_probs: Tensor of shape (G, sequence_length) with per-token log
                probabilities based on the old policy at generation time.
            response_masks: Tensor of shape (G, sequence_length) with 1s for valid
                tokens and 0s for padding.
        """
        self.prompt = prompt
        self.responses = responses
        self.rewards = rewards
        self.log_probs = log_probs
        self.response_masks = response_masks


# ===================== Base classes (no @ray.remote) =====================


class Generator:
    """Base class for text generation using TransformerLM"""

    def __init__(self, ckpt_file: str = CHECKPOINT_PATH):
        self.generator_device = get_device()
        self.generator_model: torch.nn.Module = Transformer(
            vocab_size=VOCAB_SIZE,
            num_layers=NUM_LAYERS,
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            d_ff=D_FF,
            context_length=CONTEXT_LENGTH,
            theta=THETA,
            device=self.generator_device,
        )
        self.generator_model.load_state_dict(
            torch.load(ckpt_file, map_location=self.generator_device)
        )
        self.generator_model.to(self.generator_device)
        self.generator_tokenizer = tiktoken.get_encoding("gpt2")

    @torch.no_grad()
    def generate_trajectories(
        self, prompts: list[str], verbose: bool = False
    ) -> list[Trajectory]:
        """
        Generate G responses for each prompt using TransformerLM.

        - For each prompt, generate G responses using self.model
        - Calculate log probabilities for generated tokens
        - Return list of Trajectory objects with prompts, responses, log_probs
        """
        trajs: list[Trajectory] = []

        for prompt in prompts:
            keywords = extract_keywords_from_prompt(prompt)
            if verbose:
                print(f"Prompt: {prompt} | Keywords: {keywords}")

            responses: list[torch.Tensor] = []
            log_probs_list: list[torch.Tensor] = []
            rewards: list[float] = []

            for i in range(G):
                response, log_probs = generate_response(
                    self.generator_model,
                    self.generator_tokenizer,
                    prompt,
                    SAMPLING_MAX_TOKENS,
                    SAMPLING_TEMPERATURE,
                    CONTEXT_LENGTH,
                    self.generator_device,
                )
                response_str = self.generator_tokenizer.decode(response)
                if verbose:
                    print(f"Response {i}: {response_str}")
                reward = keyword_inclusion_reward(response_str, keywords)["reward"]
                responses.append(torch.tensor(response, device=self.generator_device))
                log_probs_list.append(log_probs)
                rewards.append(reward)

            rewards_tensor = torch.tensor(rewards, device=self.generator_device)  # (G,)
            responses_tensor = torch.zeros(
                (G, SAMPLING_MAX_TOKENS), dtype=torch.long, device=self.generator_device
            )  # (G, sequence_length)
            log_probs_tensor = torch.zeros(
                (G, SAMPLING_MAX_TOKENS), device=self.generator_device
            )  # (G, sequence_length)
            response_masks = torch.zeros(
                (G, SAMPLING_MAX_TOKENS), dtype=torch.bool, device=self.generator_device
            )  # (G, sequence_length)
            for i in range(G):
                seq_len = responses[i].size(0)
                responses_tensor[i, :seq_len] = responses[i]
                log_probs_tensor[i, :seq_len] = log_probs_list[i]
                response_masks[i, :seq_len] = 1

            traj = Trajectory(
                prompt=prompt,
                responses=responses_tensor,
                rewards=rewards_tensor,
                log_probs=log_probs_tensor,
                response_masks=response_masks,
            )
            trajs.append(traj)

        return trajs

    def set_weights(self, state_dict: dict[str, Any]) -> None:
        """
        Set generator model weights.

        Args:
            state_dict: State dictionary of model weights.
        """
        self.generator_model.load_state_dict(state_dict)

    def sync_generator(self) -> None:
        if self.generator_device.type == "cuda":
            torch.cuda.synchronize(self.generator_device)


class Learner:
    """Base learner class for policy gradient updates using TransformerLM."""

    def __init__(self, ckpt_file: str = CHECKPOINT_PATH):
        """
        Args:
            ckpt_file: Path to model checkpoint.
        """
        self.learner_device = get_device()
        # Initialize the same TransformerLM model as Generator
        self.learner_model: torch.nn.Module = Transformer(
            vocab_size=VOCAB_SIZE,
            num_layers=NUM_LAYERS,
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            d_ff=D_FF,
            context_length=CONTEXT_LENGTH,
            theta=THETA,
            device=self.learner_device,
        )
        self.learner_model.load_state_dict(
            torch.load(ckpt_file, map_location=self.learner_device)
        )
        self.learner_model.to(self.learner_device)
        self.learner_optimizer: torch.optim.Optimizer = AdamW(
            self.learner_model.parameters(), **ADAMW_ARGS
        )
        self.learner_tokenizer = tiktoken.get_encoding("gpt2")

    def compute_advantages(self, trajectories: list[Trajectory]) -> torch.Tensor:
        """
        Compute advantages for GRPO.

        Args:
            trajectories (list[Trajectory]): Rollout trajectories.

        Returns:
            torch.Tensor: Advantages tensor of shape (num_trajectories, G).
        """
        # Implement the group-relative advantage computation
        # that's central to GRPO algorithm
        group_rewards = torch.stack([traj.rewards for traj in trajectories])  # (N, G)
        group_means = group_rewards.mean(dim=1, keepdim=True)  # (N, 1)
        shifted_rewards = group_rewards - group_means  # (N, G)
        if USE_STD_NORMALIZATION:
            group_stds = group_rewards.std(dim=1, keepdim=True)  # (N, 1)
            advantages = shifted_rewards / (group_stds + ADVANTAGE_EPS)  # (N, G)
        else:
            advantages = shifted_rewards
        return advantages

    def get_policy_log_probs(self, trajectories: list[Trajectory]) -> torch.Tensor:
        """
        Compute current policy log probabilities for generated tokens.

        Args:
            trajectories (list[Trajectory]): Rollout trajectories.

        Returns:
            torch.Tensor: Policy log probabilities tensor of shape
                (num_trajectories, G, sequence_length).
        """
        policy_log_probs_list: list[torch.Tensor] = []
        for traj in trajectories:
            prompt_tokens = self.learner_tokenizer.encode(traj.prompt)
            prompt_len = len(prompt_tokens)
            prompt_ids = torch.tensor(
                [prompt_tokens] * G, device=self.learner_device
            )  # (G, prompt_len)
            input_ids = torch.cat(
                [prompt_ids, traj.responses], dim=1
            )  # (G, prompt_len + seq_len)
            logits = self.learner_model(
                input_ids
            )  # (G, prompt_len + seq_len, vocab_size)
            generated_token_logits = logits[
                :, prompt_len - 1 : -1, :
            ]  # (G, seq_len, vocab_size)
            scaled_logits = generated_token_logits / SAMPLING_TEMPERATURE
            log_probs_vocab = torch.log_softmax(
                scaled_logits, dim=-1
            )  # (G, seq_len, vocab_size)
            policy_log_probs = torch.gather(
                log_probs_vocab, dim=2, index=traj.responses.unsqueeze(-1)
            ).squeeze(
                -1
            )  # (G, seq_len)
            policy_log_probs_list.append(policy_log_probs)
        return torch.stack(policy_log_probs_list)  # (N, G, sequence_length)

    def update_policy(
        self,
        trajectories: list[Trajectory],
        steps_per_rollout: int = 1,
        verbose: bool = False,
    ) -> float:
        """
        Perform one policy update step.

        Args:
            trajectories (list[Trajectory]): Rollout trajectories.
            steps_per_rollout (int): Number of training steps per rollout batch.
            verbose (bool): Whether to enable verbose logging.

        Returns:
            float: Loss value after the update step.
        """
        # Implement GRPO/PPO policy update
        # 1. Compute advantages
        # 2. Compute policy gradient loss
        # 3. Perform optimizer step
        # 4. Return loss value
        N = len(trajectories)
        advantages = self.compute_advantages(trajectories)  # (N, G)
        old_log_probs = torch.stack(
            [traj.log_probs for traj in trajectories]
        )  # (N, G, seq_len)
        response_mask = torch.stack(
            [traj.response_masks for traj in trajectories]
        )  # (N, G, seq_len)

        for step in range(steps_per_rollout):
            self.learner_optimizer.zero_grad()
            policy_log_probs = self.get_policy_log_probs(
                trajectories
            )  # (N, G, seq_len)
            loss, _ = grpo_microbatch_train_step(
                policy_log_probs=policy_log_probs.view(N * G, SAMPLING_MAX_TOKENS),
                response_mask=response_mask.view(N * G, SAMPLING_MAX_TOKENS),
                gradient_accumulation_steps=steps_per_rollout,
                loss_type=LOSS_TYPE,
                advantages=advantages.view(N * G, 1),
                old_log_probs=old_log_probs.view(N * G, SAMPLING_MAX_TOKENS),
                cliprange=CLIPRANGE,
            )
            gradient_clipping(
                list(self.learner_model.parameters()), MAX_GRAD_NORM, verbose=verbose
            )
            self.learner_optimizer.step()
            if verbose:
                print(f"Step {step}, Loss: {loss.item()}")
        return float(loss.item())

    def get_weights(self) -> dict[str, Any]:
        """
        Get current learner model weights.

        Returns:
            dict[str, Any]: State dictionary of model weights.
        """
        return self.learner_model.state_dict()

    def sync_learner(self) -> None:
        if self.learner_device.type == "cuda":
            torch.cuda.synchronize(self.learner_device)


# ===================== Combined Actor =====================


@ray.remote(num_gpus=1)
class ColocatedWorker(Generator, Learner):
    """Combined Generator and Learner in a single Ray actor."""

    def __init__(self, ckpt_file: str, steps_per_rollout: int = 1):
        Generator.__init__(self, ckpt_file=ckpt_file)
        Learner.__init__(self, ckpt_file=ckpt_file)
        self.steps_per_rollout = steps_per_rollout
        self.step_count = 0
        self.time_stats: dict[str, float] = {
            "rollout": 0.0,
            "update": 0.0,
            "transfer": 0.0,
            "total": 0.0,
        }

    def training_step(
        self, prompts: list[str], verbose: bool = False
    ) -> dict[str, Any]:
        """Perform one complete training step: generate rollout + update policy."""

        # Generate trajectories for the batch of prompts
        self.sync_generator()
        rollout_start = time.perf_counter()
        trajectories = self.generate_trajectories(prompts, verbose=verbose)
        self.sync_generator()
        rollout_end = time.perf_counter()

        # Update policy using GRPO
        self.sync_learner()
        update_start = time.perf_counter()
        loss = self.update_policy(trajectories, self.steps_per_rollout, verbose=verbose)
        self.sync_learner()
        update_end = time.perf_counter()

        # Sync weights
        transfer_start = time.perf_counter()
        self.set_weights(self.get_weights())
        self.sync_learner()
        self.sync_generator()
        transfer_end = time.perf_counter()

        self.step_count += 1

        total_time_ms = (transfer_end - rollout_start) * 1000
        rollout_time_ms = (rollout_end - rollout_start) * 1000
        update_time_ms = (update_end - update_start) * 1000
        transfer_time_ms = (transfer_end - transfer_start) * 1000
        rollout_pct = rollout_time_ms / total_time_ms * 100
        update_pct = update_time_ms / total_time_ms * 100
        transfer_pct = transfer_time_ms / total_time_ms * 100
        self.time_stats["total"] += total_time_ms
        self.time_stats["rollout"] += rollout_time_ms
        self.time_stats["update"] += update_time_ms
        self.time_stats["transfer"] += transfer_time_ms

        return {
            "step": self.step_count,
            "loss": loss,
            "num_trajectories": len(trajectories),
            "avg_reward": (
                float(torch.cat([traj.rewards for traj in trajectories]).mean())
                if trajectories
                else 0.0
            ),
            "total_time_ms": total_time_ms,
            "rollout_time_ms": f"{rollout_time_ms} ({rollout_pct:.1f}%)",
            "update_time_ms": f"{update_time_ms} ({update_pct:.1f}%)",
            "transfer_time_ms": f"{transfer_time_ms} ({transfer_pct:.1f}%)",
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get current training statistics."""
        total_time = self.time_stats["total"]
        time_stats_pct = {k: v / total_time * 100 for k, v in self.time_stats.items()}

        return {
            "step_count": self.step_count,
            "model_parameters": (
                sum(p.numel() for p in self.generator_model.parameters())
                if hasattr(self, "model")
                else 0
            ),
            "time_stats": self.time_stats,
            "time_stats_pct": time_stats_pct,
        }


# ===================== Training loop =====================


def run_training(
    keywords_file: str,
    ckpt_file: str,
    num_steps: int = 10,
    num_workers: int = 1,
    prompts_per_batch: int = 4,
    steps_per_rollout: int = 1,
    verbose: bool = False,
) -> None:
    """
    Run colocated GRPO training with text generation.

    Args:
        keywords_file (str): Path to keywords file.
        ckpt_file (str): Path to model checkpoint.
        num_steps (int): Number of training steps to perform.
        num_workers (int): Number of colocated workers to use.
        prompts_per_batch (int): Number of prompts per batch.
        steps_per_rollout (int): Number of training steps per rollout.
        verbose (bool): Whether to enable verbose logging.
    """
    # Create workers
    if num_workers != 1:
        raise ValueError("Only supports a single colocated worker")
    worker = ColocatedWorker.remote(
        ckpt_file=ckpt_file, steps_per_rollout=steps_per_rollout
    )

    # Define training prompts
    with open(keywords_file, "r") as f:
        keywords = [line.strip() for line in f.readlines() if line.strip()]

    for step in range(num_steps):
        # Sample keywords (single keyword per prompt)
        kws_batch = np.random.choice(keywords, size=prompts_per_batch, replace=False)
        prompts = [make_keyword_inclusion_prompt([kw]) for kw in kws_batch]
        # Perform training step
        step_stats = ray.get(worker.training_step.remote(prompts, verbose=verbose))
        print(f"Step {step}: {step_stats}")

    # Get final statistics
    stats: dict[str, Any] = ray.get(worker.get_statistics.remote())
    print(f"Final training statistics: {stats}")


def run_once(
    keywords_file: str,
    ckpt_file: str,
    num_steps: int = 10,
    num_workers: int = 1,
    prompts_per_batch: int = 4,
    steps_per_rollout: int = 1,
    verbose: bool = False,
) -> None:
    """Entry point for training."""
    run_training(
        keywords_file=keywords_file,
        ckpt_file=ckpt_file,
        num_steps=num_steps,
        num_workers=num_workers,
        prompts_per_batch=prompts_per_batch,
        steps_per_rollout=steps_per_rollout,
        verbose=verbose,
    )


# ===================== Entry point =====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--keywords_file", type=str, required=True, help="Path to keywords file"
    )
    parser.add_argument(
        "--ckpt_file", type=str, required=True, help="Path to model checkpoint"
    )

    parser.add_argument(
        "--steps", type=int, default=N_GRPO_STEPS, help="Number of training steps"
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of colocated workers"
    )
    parser.add_argument(
        "--prompts_per_batch", type=int, default=4, help="Number of prompts per batch"
    )
    parser.add_argument(
        "--steps_per_rollout",
        type=int,
        default=1,
        help="Number of training steps per rollout",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    def log(msg: str) -> None:
        if args.verbose:
            print(msg)

    log(f"Args: {args}.")

    ray.init(ignore_reinit_error=True)
    log("Ray initialized.")

    try:
        run_once(
            keywords_file=args.keywords_file,
            ckpt_file=args.ckpt_file,
            num_steps=args.steps,
            num_workers=args.workers,
            prompts_per_batch=args.prompts_per_batch,
            steps_per_rollout=args.steps_per_rollout,
            verbose=args.verbose,
        )
        log("Training completed.")
    finally:
        ray.shutdown()
        log("Ray shutdown.")
