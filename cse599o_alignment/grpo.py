"""
GRPO Algorithm Implementation - Assignment 3
========================================================
"""

import torch
from typing import Literal, Callable


def compute_group_normalized_reward(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalized_by_std: bool,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """
    Args:
        reward_fn: Callable[[str, str], dict[str, float]] Scores the rollout responses against
            the ground truths, producing a dict with keys "reward", "format_reward", and
            "answer_reward".

        rollout_responses: list[str] Rollouts from the policy. The length of this list is
            rollout_batch_size = n_prompts_per_rollout_batch * group_size.

        repeated_ground_truths: list[str] The ground truths for the examples. The length of this
            list is rollout_batch_size, because the ground truth for each example is repeated
            group_size times.

        group_size: int Number of responses per question (group).

        advantage_eps: float Small constant to avoid division by zero in normalization.

        normalized_by_std: bool If True, divide by the per-group standard deviation; otherwise
            subtract only the group mean.

        device: torch.device | None Device for computation and returned tensors.

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]].

        advantages: shape (rollout_batch_size,). Group-normalized rewards for each rollout
            response.

        raw_rewards: shape (rollout_batch_size,). Unnormalized rewards for each rollout
            response.

        metadata: your choice of other statistics to log (e.g. mean, std, max/min of rewards).
    """
    if len(rollout_responses) % group_size != 0:
        raise ValueError("rollout_responses length must be divisible by group_size")
    n_prompts_per_rollout_batch = len(rollout_responses) // group_size

    rewards: list[float] = []
    for response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        reward_dict = reward_fn(response, ground_truth)
        reward = reward_dict["reward"]
        rewards.append(reward)
    raw_rewards = torch.tensor(rewards, device=device)

    grouped_rewards = raw_rewards.view(n_prompts_per_rollout_batch, group_size)
    group_means = grouped_rewards.mean(dim=1, keepdim=True)
    shifted_rewards = grouped_rewards - group_means
    if normalized_by_std:
        group_stds = grouped_rewards.std(dim=1, keepdim=True)
        normalized_rewards = shifted_rewards / (group_stds + advantage_eps)
    else:
        normalized_rewards = shifted_rewards
    advantages = normalized_rewards.view(-1)
    metadata: dict[str, float] = {}
    return advantages, raw_rewards, metadata


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute GRPO loss with PPO-style clipping for training stability.

    Args:
        advantages: torch.Tensor Shape (batch_size, 1), per-example advantages A.

        policy_log_probs: torch.Tensor Shape (batch_size, sequence_length), per-token log
        probs from the policy being trained.

        old_log_probs: torch.Tensor Shape (batch_size, sequence_length), per-token log probs
        from the old policy.

        cliprange: float Clip parameter ϵ (e.g. 0.2).

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].

        loss: torch.Tensor of shape (batch_size, sequence_length), the per-token clipped
        loss.

        metadata: dict containing whatever you want to log. We suggest logging whether each
        token was clipped or not, i.e., whether the clipped policy gradient loss on the RHS of
        the min was lower than the LHS.
    """
    prob_ratio = torch.exp(policy_log_probs - old_log_probs)
    unclipped = prob_ratio * advantages
    clipped_ratio = torch.clamp(prob_ratio, 1.0 - cliprange, 1.0 + cliprange)
    clipped = clipped_ratio * advantages
    is_clipped = (prob_ratio < (1.0 - cliprange)) | (prob_ratio > (1.0 + cliprange))
    loss = -torch.min(unclipped, clipped)
    metadata = {"is_clipped": is_clipped}
    return loss, metadata


def masked_mean(
    tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None
) -> torch.Tensor:
    """
    Compute mean of tensor elements where mask is True.

    Args:
        tensor: torch.Tensor The data to be averaged.

        mask: torch.Tensor Same shape as tensor; positions with 1 are included in the mean.

        dim: int | None Dimension over which to average. If None, compute the mean over all
        masked elements.

    Returns:
        torch.Tensor The masked mean; shape matches tensor.mean(dim) semantics.
    """
    if mask.dtype != torch.bool:
        mask = mask == 1
    masked = tensor.masked_fill(~mask, 0)

    total = torch.sum(masked, dim=dim)
    count = torch.sum(mask, dim=dim)
    mean = total / count
    return mean


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal[
        "grpo_clip"
    ],  # for this assignment, only "grpo_clip" is required
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Execute one GRPO training microbatch step with gradient accumulation.

    Args:
        policy_log_probs: (batch_size, sequence_length), per-token log-probabilities from the
            policy being trained.

        response_mask: (batch_size, sequence_length), 1 for response tokens, 0 for
            prompt/padding.

        gradient_accumulation_steps: Number of microbatches per optimizer step.

        loss_type: "grpo_clip".

        raw_rewards: Needed when loss_type == "no_baseline"; shape (batch_size, 1).

        advantages: Needed when loss_type != "no_baseline"; shape (batch_size, 1).

        old_log_probs: Required for GRPO-Clip; shape (batch_size, sequence_length).

        cliprange: Clip parameter ϵ for GRPO-Clip.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].

        loss: scalar tensor. The microbatch loss, adjusted for gradient accumulation. We return
            this so we can log it.

        metadata: Dict with metadata from the underlying loss call, and any other statistics you
            might want to log.
    """
    if loss_type != "grpo_clip":
        raise ValueError(f"Unsupported loss_type: {loss_type}")
    if advantages is None:
        raise ValueError("advantages must be provided for GRPO-Clip loss")
    if old_log_probs is None:
        raise ValueError("old_log_probs must be provided for GRPO-Clip loss")
    if cliprange is None:
        raise ValueError("cliprange must be provided for GRPO-Clip loss")

    token_loss, loss_metadata = compute_grpo_clip_loss(
        advantages, policy_log_probs, old_log_probs, cliprange
    )
    masked_loss = masked_mean(token_loss, response_mask, dim=1)  # (batch_size,)
    microbatch_loss = torch.mean(masked_loss) / gradient_accumulation_steps
    microbatch_loss.backward()
    return microbatch_loss, loss_metadata
