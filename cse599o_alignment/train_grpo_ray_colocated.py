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
import json
import os
import ray
import torch
import tiktoken
import time
from typing import Any
import numpy as np

from cse599o_basics.model import Transformer
from cse599o_basics.optimizer import AdamW, gradient_clipping
from cse599o_alignment.grpo import grpo_microbatch_train_step, kl_divergence
from cse599o_alignment.plot_util import plot_kl_rewards
from cse599o_alignment.train_util import (
    Prompt,
    extract_keywords_from_prompt,
    generate_response,
    generate_response_with_probs,
    keyword_inclusion_reward,
    make_keyword_inclusion_prompt,
    set_seed,
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
        # Lazily computed input IDs (prompt + responses)
        self._input_ids: torch.Tensor | None = None  # (G, prompt_len + sequence_length)

    def get_input_ids(
        self, tokenizer: tiktoken.Encoding, device: torch.device
    ) -> torch.Tensor:
        """
        Get input IDs combining prompt and responses.

        Args:
            tokenizer: Tokenizer to encode the prompt.
            device: Device for the returned tensor.

        Returns:
            torch.Tensor: Input IDs tensor of shape (G, prompt_len + sequence_length).
        """
        if self._input_ids is None:
            prompt_tokens = tokenizer.encode(self.prompt)
            prompt_ids = torch.tensor(
                [prompt_tokens] * G, device=device
            )  # (G, prompt_len)
            self._input_ids = torch.cat(
                [prompt_ids, self.responses], dim=1
            )  # (G, prompt_len + sequence_length)

        return torch.clone(self._input_ids.detach()).to(device)


def compute_log_probs(
    model: torch.nn.Module,
    tokenizer: tiktoken.Encoding,
    trajectories: list[Trajectory],
    device: torch.device,
) -> torch.Tensor:
    """
    Compute log probabilities for generated tokens in trajectories.

    Args:
        model (torch.nn.Module): The language model for computing log probabilities.
        tokenizer (tiktoken.Encoding): The tokenizer for encoding/decoding text.
        trajectories (list[Trajectory]): List of Trajectory objects.
        device (torch.device): Device for computation.

    Returns:
        torch.Tensor: Log probabilities tensor of shape
            (num_trajectories, G, sequence_length).
    """
    log_probs_list: list[torch.Tensor] = []
    for traj in trajectories:
        input_ids = traj.get_input_ids(tokenizer, device)  # (G, prompt_len + seq_len)
        prompt_len = input_ids.shape[1] - SAMPLING_MAX_TOKENS
        logits = model(input_ids)  # (G, prompt_len + seq_len, vocab_size)
        generated_token_logits = logits[
            :, prompt_len - 1 : -1, :
        ]  # (G, seq_len, vocab_size)
        scaled_logits = generated_token_logits / SAMPLING_TEMPERATURE
        log_probs_vocab = torch.log_softmax(
            scaled_logits, dim=-1
        )  # (G, seq_len, vocab_size)
        log_probs = torch.gather(
            log_probs_vocab, dim=2, index=traj.responses.unsqueeze(-1)
        ).squeeze(
            -1
        )  # (G, seq_len)
        log_probs_list.append(log_probs)
    return torch.stack(log_probs_list)  # (N, G, seq_len)


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
                response, log_probs = generate_response_with_probs(
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

        self.learner_stats: dict[str, Any] = {
            "avg_train_rewards": [],
            "losses": [],
            "responses": [],
            "avg_val_rewards": [],
            "ref_kls": [],
            "old_kls": [],
        }

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
        policy_log_probs = compute_log_probs(
            self.learner_model,
            self.learner_tokenizer,
            trajectories,
            self.learner_device,
        )  # (N, G, seq_len)
        return policy_log_probs

    @torch.no_grad()
    def get_val_reward(self, val_prompts: list[Prompt]) -> float:
        """
        Compute average reward on validation prompts.

        Args:
            val_prompts (list[Prompt]): List of validation prompts.

        Returns:
            float: Average reward over validation prompts.
        """
        total_reward = 0.0
        responses: list[str] = []

        for prompt in val_prompts:
            response = generate_response(
                model = self.learner_model,
                tokenizer = self.learner_tokenizer,
                prompt = prompt.token_tensor,
                max_tokens = SAMPLING_MAX_TOKENS,
                temperature = SAMPLING_TEMPERATURE,
                context_length = CONTEXT_LENGTH,
                device = self.learner_device,
            )
            reward = keyword_inclusion_reward(response, prompt.keywords)["reward"]
            total_reward += reward
            responses.append(response)

        self.learner_stats["responses"].append(responses)
        avg_reward = total_reward / len(val_prompts)
        return avg_reward

    def update_policy(
        self,
        trajectories: list[Trajectory],
        steps_per_rollout: int = 1,
        monitor_kl: bool = False,
        ref_log_probs: torch.Tensor | None = None,
        val_prompts: list[Prompt] | None = None,
        verbose: bool = False,
    ) -> float:
        """
        Perform one policy update step.

        Args:
            trajectories (list[Trajectory]): Rollout trajectories.
            steps_per_rollout (int): Number of training steps per rollout batch.
            monitor_kl (bool): Whether to monitor KL divergence.
            ref_log_probs (torch.Tensor | None): Reference log probabilities for KL computation.
                Required if monitor_kl is True. Shape (num_trajectories, G, sequence_length).
            val_prompts (list[Prompt] | None): Validation prompts. Required if monitor_kl is True.
            verbose (bool): Whether to enable verbose logging.

        Returns:
            float: Loss value after the update step.
        """
        # Implement GRPO/PPO policy update
        # 1. Compute advantages
        # 2. Compute policy gradient loss
        # 3. Perform optimizer step
        # 4. Return loss value
        if monitor_kl and ref_log_probs is None:
            raise ValueError("ref_log_probs must be provided when monitor_kl is True")
        if monitor_kl and val_prompts is None:
            raise ValueError("val_prompts must be provided when monitor_kl is True")

        N = len(trajectories)
        advantages = self.compute_advantages(trajectories)  # (N, G)
        old_log_probs = torch.stack(
            [traj.log_probs for traj in trajectories]
        )  # (N, G, seq_len)
        response_mask = torch.stack(
            [traj.response_masks for traj in trajectories]
        )  # (N, G, seq_len)

        total_loss = 0.0
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

            total_loss += loss.item()
            if verbose:
                print(f"Microstep {step}, Loss: {loss.item()}")

        if monitor_kl:
            final_policy_log_probs = self.get_policy_log_probs(
                trajectories
            )  # (N, G, seq_len)
            kl_ref = kl_divergence(
                final_policy_log_probs.view(N * G, SAMPLING_MAX_TOKENS),
                ref_log_probs.view(N * G, SAMPLING_MAX_TOKENS),
                response_mask.view(N * G, SAMPLING_MAX_TOKENS),
            ).item()
            kl_old = kl_divergence(
                final_policy_log_probs.view(N * G, SAMPLING_MAX_TOKENS),
                old_log_probs.view(N * G, SAMPLING_MAX_TOKENS),
                response_mask.view(N * G, SAMPLING_MAX_TOKENS),
            ).item()
            print(f"KL divergence w.r.t reference: {kl_ref}")
            print(f"KL divergence w.r.t old policy: {kl_old}")
            
            avg_val_reward = self.get_val_reward(val_prompts)
            print(f"Average validation reward: {avg_val_reward}")

            self.learner_stats["avg_val_rewards"].append(avg_val_reward)
            self.learner_stats["ref_kls"].append(kl_ref)
            self.learner_stats["old_kls"].append(kl_old)

        avg_loss = total_loss / steps_per_rollout
        avg_reward = float(torch.cat([traj.rewards for traj in trajectories]).mean())
        self.learner_stats["losses"].append(avg_loss)
        self.learner_stats["avg_train_rewards"].append(avg_reward)
        return avg_loss

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

    def export_learner_stats(self, results_dir: str) -> None:
        """Export learner validation statistics to files."""
        os.makedirs(results_dir, exist_ok=True)

        responses_file = os.path.join(results_dir, "val_responses.json")
        with open(responses_file, "w") as f:
            json.dump(self.learner_stats["responses"], f, indent=4)

        stats_file = os.path.join(results_dir, "stats.json")
        with open(stats_file, "w") as f:
            json.dump(
                {
                    "losses": self.learner_stats["losses"],
                    "avg_train_rewards": self.learner_stats["avg_train_rewards"],
                    "avg_val_rewards": self.learner_stats["avg_val_rewards"],
                    "ref_kls": self.learner_stats["ref_kls"],
                    "old_kls": self.learner_stats["old_kls"],
                },
                f,
                indent=4,
            )
        
        plot_file = os.path.join(results_dir, "kl_rewards.png")
        plot_kl_rewards(
            ref_kls=self.learner_stats["ref_kls"],
            old_kls=self.learner_stats["old_kls"],
            avg_train_rewards=self.learner_stats["avg_train_rewards"],
            avg_val_rewards=self.learner_stats["avg_val_rewards"],
            out_file=plot_file,
        )


class ReferenceModel:
    """Reference model for KL divergence computation in GRPO."""

    def __init__(self, ckpt_file: str = CHECKPOINT_PATH):
        self.ref_device = get_device()
        self.ref_model: torch.nn.Module = Transformer(
            vocab_size=VOCAB_SIZE,
            num_layers=NUM_LAYERS,
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            d_ff=D_FF,
            context_length=CONTEXT_LENGTH,
            theta=THETA,
            device=self.ref_device,
        )
        self.ref_model.load_state_dict(
            torch.load(ckpt_file, map_location=self.ref_device)
        )
        self.ref_model.to(self.ref_device)
        self.ref_tokenizer = tiktoken.get_encoding("gpt2")

    @torch.no_grad()
    def get_reference_log_probs(self, trajectories: list[Trajectory]) -> torch.Tensor:
        """
        Compute reference log probabilities for generated tokens.

        Args:
            trajectories (list[Trajectory]): Rollout trajectories.

        Returns:
            torch.Tensor: Reference log probabilities tensor of shape
                (num_trajectories, G, sequence_length).
        """
        ref_log_probs = compute_log_probs(
            self.ref_model,
            self.ref_tokenizer,
            trajectories,
            self.ref_device,
        )  # (N, G, seq_len)
        return ref_log_probs

    def sync_reference(self) -> None:
        if self.ref_device.type == "cuda":
            torch.cuda.synchronize(self.ref_device)


# ===================== Combined Actor =====================


@ray.remote(num_gpus=1)
class ColocatedWorker(Generator, Learner, ReferenceModel):
    """Combined Generator and Learner in a single Ray actor."""

    def __init__(self, ckpt_file: str, prompts_val: list[str], steps_per_rollout: int = 1):
        Generator.__init__(self, ckpt_file=ckpt_file)
        Learner.__init__(self, ckpt_file=ckpt_file)
        ReferenceModel.__init__(self, ckpt_file=ckpt_file)

        self.steps_per_rollout = steps_per_rollout
        self.step_count = 0
        self.time_stats: dict[str, float] = {
            "rollout": 0.0,
            "update": 0.0,
            "transfer": 0.0,
            "reference": 0.0,
            "total": 0.0,
        }

        self.prompts_val = [Prompt(p, self.learner_tokenizer, self.learner_device) for p in prompts_val]
        set_seed()

    def training_step(
        self, prompts: list[str], monitor_kl: bool = False, verbose: bool = False
    ) -> dict[str, Any]:
        """Perform one complete training step: generate rollout + update policy."""

        # Generate trajectories for the batch of prompts
        self.sync_generator()
        rollout_start = time.perf_counter()
        trajectories = self.generate_trajectories(prompts, verbose=verbose)
        self.sync_generator()
        rollout_end = time.perf_counter()

        ref_log_probs: torch.Tensor | None = None
        ref_time = 0.0
        if monitor_kl:
            self.sync_reference()
            ref_start = time.perf_counter()
            ref_log_probs = self.get_reference_log_probs(trajectories)
            self.sync_reference()
            ref_end = time.perf_counter()
            ref_time = (ref_end - ref_start) * 1000

        # Update policy using GRPO
        self.sync_learner()
        update_start = time.perf_counter()
        loss = self.update_policy(
            trajectories,
            self.steps_per_rollout,
            monitor_kl=monitor_kl,
            ref_log_probs=ref_log_probs,
            val_prompts=self.prompts_val,
            verbose=verbose,
        )
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
        ref_pct = ref_time / total_time_ms * 100
        self.time_stats["total"] += total_time_ms
        self.time_stats["rollout"] += rollout_time_ms
        self.time_stats["update"] += update_time_ms
        self.time_stats["transfer"] += transfer_time_ms
        self.time_stats["reference"] += ref_time

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
            "reference_time_ms": f"{ref_time} ({ref_pct:.1f}%)",
        }

    def get_statistics(self, result_dir: str) -> dict[str, Any]:
        """Get current training statistics."""
        self.export_learner_stats(result_dir)

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

    def save_ckpt(self, result_dir: str, step: int) -> None:
        """
        Save current learner model checkpoint.

        Args:
            result_dir (str): Directory to save checkpoint.
            step (int): Number of training steps completed.
        """
        os.makedirs(result_dir, exist_ok=True)
        ckpt_file = os.path.join(result_dir, f"step{step}.pt")
        model_state_dict = self.learner_model.state_dict()
        optimizer_states = self.learner_optimizer.state_dict()
        curr_iter = self.step_count
        checkpoint = {
            "model_state_dict": model_state_dict,
            "optimizer_states": optimizer_states,
            "iteration": curr_iter,
        }
        torch.save(checkpoint, ckpt_file)


# ===================== Training loop =====================


def run_training(
    keywords_file: str,
    ckpt_file: str,
    result_dir: str,
    num_steps: int = 10,
    num_workers: int = 1,
    prompts_per_batch: int = 32,
    num_val_prompts: int = 32,
    steps_per_rollout: int = 1,
    ckpt_interval: int = 5,
    monitor_kl: bool = False,
    verbose: bool = False,
) -> None:
    """
    Run colocated GRPO training with text generation.

    Args:
        keywords_file (str): Path to keywords file.
        ckpt_file (str): Path to model checkpoint.
        result_dir (str): Directory to save results.
        num_steps (int): Number of training steps to perform.
        num_workers (int): Number of colocated workers to use.
        prompts_per_batch (int): Number of prompts per batch.
        num_val_prompts (int): Number of validation prompts.
        steps_per_rollout (int): Number of training steps per rollout.
        monitor_kl (bool): Whether to monitor KL divergence.
        verbose (bool): Whether to enable verbose logging.
    """
    if num_workers != 1:
        raise ValueError("Only supports a single colocated worker")

    # Define training prompts
    with open(keywords_file, "r") as f:
        keywords = [line.strip() for line in f.readlines() if line.strip()]
    # Last num_val_prompts keywords are used for validation
    if num_val_prompts + prompts_per_batch > len(keywords):
        raise ValueError("Not enough keywords for training and validation")
    keywords_train = keywords[: -num_val_prompts]
    keywords_val = keywords[-num_val_prompts :]
    prompts_val = [make_keyword_inclusion_prompt([kw]) for kw in keywords_val]

    # Create workers
    worker = ColocatedWorker.remote(
        ckpt_file=ckpt_file, steps_per_rollout=steps_per_rollout, prompts_val=prompts_val
    )

    set_seed()
    for step in range(num_steps):
        # Sample keywords (single keyword per prompt)
        kws_batch = np.random.choice(keywords_train, size=prompts_per_batch, replace=False)
        prompts_train = [make_keyword_inclusion_prompt([kw]) for kw in kws_batch]
        # Perform training step
        step_stats = ray.get(
            worker.training_step.remote(prompts_train, monitor_kl=monitor_kl, verbose=verbose)
        )
        print(f"Step {step}: {step_stats}")
        if (step + 1) % ckpt_interval == 0:
            ray.get(worker.save_ckpt.remote(result_dir, step + 1))

    # Get final statistics
    stats: dict[str, Any] = ray.get(worker.get_statistics.remote(result_dir))
    print(f"Final training statistics: {stats}")


def run_once(
    keywords_file: str,
    ckpt_file: str,
    result_dir: str,
    num_steps: int = 10,
    num_workers: int = 1,
    prompts_per_batch: int = 32,
    num_val_prompts: int = 32,
    steps_per_rollout: int = 1,
    ckpt_interval: int = 5,
    monitor_kl: bool = False,
    verbose: bool = False,
) -> None:
    """Entry point for training."""
    run_training(
        keywords_file=keywords_file,
        ckpt_file=ckpt_file,
        result_dir=result_dir,
        num_steps=num_steps,
        num_workers=num_workers,
        prompts_per_batch=prompts_per_batch,
        num_val_prompts=num_val_prompts,
        steps_per_rollout=steps_per_rollout,
        ckpt_interval=ckpt_interval,
        monitor_kl=monitor_kl,
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
        "--result_dir", type=str, required=True, help="Directory to save results"
    )

    parser.add_argument(
        "--steps", type=int, default=N_GRPO_STEPS, help="Number of training steps"
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of colocated workers"
    )
    parser.add_argument(
        "--prompts_per_batch", type=int, default=32, help="Number of prompts per batch"
    )
    parser.add_argument(
        "--num_val_prompts", type=int, default=32, help="Number of validation prompts"
    )
    parser.add_argument(
        "--steps_per_rollout",
        type=int,
        default=1,
        help="Number of training steps per rollout",
    )
    parser.add_argument(
        "--ckpt_interval", type=int, default=5, help="Checkpoint save interval"
    )
    parser.add_argument(
        "--monitor_kl", action="store_true", help="Monitor KL divergence"
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
        run_once(
            keywords_file=args.keywords_file,
            ckpt_file=args.ckpt_file,
            result_dir=args.result_dir,
            num_steps=args.steps,
            num_workers=args.workers,
            prompts_per_batch=args.prompts_per_batch,
            num_val_prompts=args.num_val_prompts,
            steps_per_rollout=args.steps_per_rollout,
            ckpt_interval=args.ckpt_interval,
            monitor_kl=args.monitor_kl,
            verbose=args.verbose,
        )
        log("Training completed.")
    finally:
        ray.shutdown()
        log("Ray shutdown.")
