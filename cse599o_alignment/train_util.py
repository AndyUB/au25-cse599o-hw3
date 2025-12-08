import tiktoken
import numpy as np
import torch
from torch.distributions import Categorical


KEYWORD_INCLUSION_PROMPT_PREFIX = "Write a story that includes the word "
KEYWORD_INCLUSION_PROMPT_SUFFIX = ". One day, "


def make_keyword_inclusion_prompt(keywords: list[str]) -> str:
    """
    Create a prompt string that asks for inclusion of the specified keywords.

    Args:
        keywords (list[str]): List of keywords to include in the prompt.

    Returns:
        str: The formatted prompt string.
    """
    if len(keywords) == 0:
        raise ValueError("keywords list must not be empty")

    kw_str = ", ".join(keywords)
    prompt = KEYWORD_INCLUSION_PROMPT_PREFIX + kw_str + KEYWORD_INCLUSION_PROMPT_SUFFIX
    return prompt


def extract_keywords_from_prompt(prompt: str) -> list[str]:
    """
    Extract keywords from a formatted prompt string.

    Args:
        prompt (str): The prompt string.

    Returns:
        list[str]: List of extracted keywords.
    """
    if not prompt.startswith(KEYWORD_INCLUSION_PROMPT_PREFIX) or not prompt.endswith(
        KEYWORD_INCLUSION_PROMPT_SUFFIX
    ):
        raise ValueError(f"Prompt format is incorrect: {prompt}")

    kw_str = prompt[
        len(KEYWORD_INCLUSION_PROMPT_PREFIX) : -len(KEYWORD_INCLUSION_PROMPT_SUFFIX)
    ]
    keywords = kw_str.split(", ")
    return keywords


class Prompt:
    def __init__(self, text: str, tokenizer: tiktoken.Encoding, device: torch.device):
        self.text = text
        self.keywords = extract_keywords_from_prompt(text)
        self.tokens = tokenizer.encode(text)
        self.token_tensor = torch.tensor(
            [self.tokens], device=device
        )  # (1, prompt_len)


def keyword_inclusion_reward(
    response: str,
    keywords: list[str],
) -> dict[str, float]:
    """
    Compute rewards based on case-insensitive keyword inclusion in the response.
    If all keywords are included, the reward is 1; otherwise, 0.

    Args:
        response (str): The generated response text.
        keywords (list[str]): List of keywords to check for inclusion.

    Returns:
        dict[str, float]: Dictionary with keys "reward", "format_reward",
            and "answer_reward" and corresponding reward values.
    """
    if len(keywords) == 0:
        raise ValueError("keywords list must not be empty")

    response = response.lower()
    if all(keyword.lower() in response for keyword in keywords):
        answer_reward = 1.0
    else:
        answer_reward = 0.0
    return {
        "reward": answer_reward,
        "format_reward": 0.0,
        "answer_reward": answer_reward,
    }


def set_seed(seed: int = 599) -> None:
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): The seed value to set.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


@torch.no_grad()
def generate_response_with_probs(
    model: torch.nn.Module,
    tokenizer: tiktoken.Encoding,
    prompt: str,
    max_tokens: int,
    temperature: float,
    context_length: int,
    device: torch.device,
) -> tuple[list[int], torch.Tensor]:
    """
    Generate a response with log probabilities from the model given a prompt.

    Args:
        model (torch.nn.Module): The language model for generation.
        tokenizer (tiktoken.Encoding): The tokenizer for encoding/decoding text.
        prompt (str): The input prompt string.
        max_tokens (int): Maximum number of tokens to generate.
        temperature (float): Sampling temperature.
        context_length (int): Maximum context length for the model.
        device (torch.device): Device for computation.

    Returns:
        tuple[list[int], torch.Tensor]: Generated response tokens
            and log probabilities.
    """
    model.eval()
    prompt_tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([prompt_tokens], device=device)  # (1, seq_len)
    generated_tokens: list[int] = []
    log_probs: list[float] = []

    for _ in range(max_tokens):
        if input_ids.shape[1] > context_length:
            break

        logits = model(input_ids)  # (1, seq_len, vocab_size)
        next_token_logits = logits[:, -1, :] / temperature  # (1, vocab_size)
        dist = Categorical(logits=next_token_logits)
        next_token_id = dist.sample()  # (1,)
        log_prob: torch.Tensor = dist.log_prob(next_token_id)

        generated_tokens.append(int(next_token_id.item()))
        log_probs.append(log_prob.item())

        if next_token_id.item() == tokenizer.eot_token:
            break

        input_ids = torch.cat(
            [input_ids, next_token_id.unsqueeze(0)], dim=1
        )  # (1, seq_len+1)

    probs_tensor = torch.tensor(log_probs, device=device)  # (generated_seq_len,)
    return generated_tokens, probs_tensor


@torch.no_grad()
def batch_generate_responses(
    model: torch.nn.Module,
    tokenizer: tiktoken.Encoding,
    prompts: list[str],
    group_size: int,
    max_tokens: int,
    temperature: float,
    context_length: int,
    device: torch.device,
    with_probs: bool = True,
    profile: bool = False,
) -> tuple[list[list[int]], list[list[float]] | None]:
    """
    Generate a batch of responses from the model given a list of prompts.

    Args:
        model (torch.nn.Module): The language model for generation.
        tokenizer (tiktoken.Encoding): The tokenizer for encoding/decoding text.
        prompts (list[str]): Input prompts.
        group_size (int): Number of rollouts per prompt.
        max_tokens (int): Maximum number of tokens to generate.
        temperature (float): Sampling temperature.
        context_length (int): Maximum context length for the model.
        device (torch.device): Device for computation.
        with_probs (bool): Whether to return log probabilities.
        profile (bool): Whether to enable profiling.

    Returns:
        tuple[list[list[int]], list[list[float]] | None]: Generated response tokens,
            and log probabilities if with_probs is True, otherwise None.
    """
    num_prompts = len(prompts)
    if num_prompts == 0:
        raise ValueError("prompts list must not be empty")
    num_rollouts = num_prompts * group_size

    prompt_tokens_list = []
    for prompt in prompts:
        prompt_tokens = tokenizer.encode(prompt)
        prompt_tokens_list.extend([prompt_tokens] * group_size)
    prompt_lens = [len(tokens) for tokens in prompt_tokens_list]
    max_prompt_len = max(prompt_lens)
    next_token_positions = (
        torch.tensor(prompt_lens, device=device, dtype=torch.long) - 1
    )  # (num_rollouts,)
    input_ids = torch.zeros(
        (num_rollouts, max_prompt_len), dtype=torch.long, device=device
    )
    for i, tokens in enumerate(prompt_tokens_list):
        input_ids[i, : len(tokens)] = torch.tensor(tokens, device=device)
    rollout_ids = torch.arange(num_rollouts, dtype=torch.long, device=device)

    model.eval()
    generated_tokens: list[list[int]] = [[] for _ in range(num_rollouts)]
    log_probs: list[list[float]] | None = None
    if with_probs:
        log_probs = [[] for _ in range(num_rollouts)]

    # prev num_valid == num_rollouts
    for _ in range(max_tokens):
        # input_ids: (prev num_valid, prev seq_len -> seq_len)
        if input_ids.shape[1] > context_length:
            if profile:
                raise RuntimeError("Input exceeds context length during profiling")

            discard_inp_result = discard_long_inputs(
                input_ids, context_length, next_token_positions, rollout_ids
            )
            if discard_inp_result is None:
                break
            # (num_valid, seq_len), (num_valid,), (num_valid,)
            input_ids, next_token_positions, rollout_ids = discard_inp_result
        # else: num_valid == prev num_valid

        num_valid = input_ids.shape[0]
        logits: torch.Tensor = model(input_ids)  # (num_valid, seq_len, vocab_size)
        batch_dim_idx = torch.arange(num_valid, device=device)  # (num_valid,)
        next_token_logits = (
            logits[batch_dim_idx, next_token_positions, :] / temperature
        )  # (num_valid, vocab_size)
        dist = Categorical(logits=next_token_logits)
        next_token_ids = dist.sample()  # (num_valid,)
        next_token_probs: torch.Tensor | None = None
        if with_probs:
            next_token_probs = dist.log_prob(next_token_ids)  # (num_valid,)

        for i in range(num_valid):
            rollout_id = int(rollout_ids[i].item())
            next_token_id = int(next_token_ids[i].item())
            generated_tokens[rollout_id].append(next_token_id)

            if with_probs:
                next_token_prob = float(next_token_probs[i].item())
                log_probs[rollout_id].append(next_token_prob)

        next_token_positions = next_token_positions + 1  # (new num_valid,)
        pad_column = torch.zeros((num_valid, 1), dtype=torch.long, device=device)
        input_ids = torch.cat([input_ids, pad_column], dim=1)  # (num_valid, seq_len+1)
        input_ids[batch_dim_idx, next_token_positions] = next_token_ids

        if profile:
            continue

        discard_oup_result = discard_eot_outputs(
            input_ids, tokenizer.eot_token, next_token_positions, rollout_ids
        )
        if discard_oup_result is None:
            break
        # (new num_valid, seq_len+1), (new num_valid,), (new num_valid,)
        input_ids, next_token_positions, rollout_ids = discard_oup_result

    return generated_tokens, log_probs


def discard_long_inputs(
    input_ids: torch.Tensor,
    context_length: int,
    next_token_positions: torch.Tensor,
    rollout_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """
    Discard inputs that exceed the context length.

    Args:
        input_ids (torch.Tensor): Input tensor of shape (batch_size, seq_len).
            batch_size is the number of rollouts in the batch.
        context_length (int): Maximum context length for the model.
        next_token_positions (torch.Tensor): Tensor of shape (batch_size,)
            indicating the position of the next token to generate.
        rollout_ids (torch.Tensor): Tensor of shape (batch_size,) indicating the
            rollout IDs corresponding to each input.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None: Filtered input_ids
            of shape (num_valid, max_next_token_position + 1), next_token_positions
            of shape (num_valid,), and rollout_ids of shape (num_valid,); or None
            if no valid inputs remain.
    """
    # e.g., input=[0,1], output=[1,2], next_token_position=1 (index in output),
    # context_length=2 -> valid (next_token_position == context_length - 1)
    # context_length=1 -> invalid (next_token_position == context_length)
    valid_mask = next_token_positions < context_length
    filtered_input_ids = input_ids[valid_mask]  # (num_valid, seq_len)

    if filtered_input_ids.shape[0] == 0:
        return None

    filtered_next_token_positions = next_token_positions[valid_mask]  # (num_valid,)
    filtered_rollout_ids = rollout_ids[valid_mask]  # (num_valid,)
    max_next_token_position = torch.max(filtered_next_token_positions)
    # input and output has same leading dimensions, so
    # last index in input_ids must be at least max_next_token_position
    filtered_input_ids = filtered_input_ids[
        :, : max_next_token_position + 1
    ]  # (num_valid, max_next_token_position+1)
    return filtered_input_ids, filtered_next_token_positions, filtered_rollout_ids


def discard_eot_outputs(
    sequence_ids: torch.Tensor,
    eot_token_id: int,
    curr_token_positions: torch.Tensor,
    rollout_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """
    Discard outputs that have generated the end-of-text (EOT) token.

    Args:
        sequence_ids (torch.Tensor): Tensor of shape (batch_size, seq_len)
            containing generated token IDs.
        eot_token_id (int): The end-of-text (EOT) token ID.
        curr_token_positions (torch.Tensor): Tensor of shape (batch_size,)
            indicating the current token positions. The current token is the
            one that was just generated.
        rollout_ids (torch.Tensor): Tensor of shape (batch_size,) indicating the
            rollout IDs corresponding to each output.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None: Filtered
            sequence_ids of shape (num_valid, seq_len), curr_token_positions
            of shape (num_valid,), and rollout_ids of shape (num_valid,); or
            None if no valid outputs remain.
    """
    # curr_token_positions points to the just generated token
    batch_dim_idx = torch.arange(sequence_ids.shape[0], device=sequence_ids.device)
    curr_token_ids = sequence_ids[batch_dim_idx, curr_token_positions]  # (batch_size,)
    valid_mask = curr_token_ids != eot_token_id  # (batch_size,)
    if torch.all(valid_mask):
        return sequence_ids, curr_token_positions, rollout_ids

    filtered_sequence_ids = sequence_ids[valid_mask]  # (num_valid, seq_len)
    if filtered_sequence_ids.shape[0] == 0:
        return None

    filtered_token_positions = curr_token_positions[valid_mask]  # (num_valid,)
    filtered_rollout_ids = rollout_ids[valid_mask]  # (num_valid,)
    return filtered_sequence_ids, filtered_token_positions, filtered_rollout_ids


@torch.no_grad()
def generate_response(
    model: torch.nn.Module,
    tokenizer: tiktoken.Encoding,
    prompt: torch.Tensor,
    max_tokens: int,
    temperature: float,
    context_length: int,
    device: torch.device,
) -> str:
    """
    Generate a response from the model given a prompt.

    Args:
        model (torch.nn.Module): The language model for generation.
        tokenizer (tiktoken.Encoding): The tokenizer for encoding/decoding text.
        prompt (torch.Tensor): The input prompt tensor of shape (1, prompt_length).
        max_tokens (int): Maximum number of tokens to generate.
        temperature (float): Sampling temperature.
        context_length (int): Maximum context length for the model.
        device (torch.device): Device for computation.

    Returns:
        str: The generated response string.
    """
    model.eval()
    input_ids = prompt.to(device)  # (1, seq_len)
    generated_tokens: list[int] = []

    for _ in range(max_tokens):
        if input_ids.shape[1] > context_length:
            break

        logits = model(input_ids)  # (1, seq_len, vocab_size)
        next_token_logits = logits[:, -1, :] / temperature  # (1, vocab_size)
        dist = Categorical(logits=next_token_logits)
        next_token_id = dist.sample()  # (1,)
        generated_tokens.append(int(next_token_id.item()))

        if next_token_id.item() == tokenizer.eot_token:
            break

        input_ids = torch.cat(
            [input_ids, next_token_id.unsqueeze(0)], dim=1
        )  # (1, seq_len+1)

    response = tokenizer.decode(generated_tokens)
    return response


def group_decode_with_rewards(
    tokenizer: tiktoken.Encoding,
    prompt: str,
    grouped_tokens: list[list[int]],
) -> tuple[list[str], list[float], list[str]]:
    """
    Decode generated tokens for a rollout group and compute rewards.

    Args:
        tokenizer (tiktoken.Encoding): The tokenizer for encoding/decoding text.
        prompt (str): The input prompt string.
        grouped_tokens (list[list[int]]): List of generated token lists for each
            rollout in the group.

    Returns:
        tuple[list[str], list[float], list[str]]: Decoded responses,
            rewards for each response, and keywords extracted from the prompt.
    """
    keywords = extract_keywords_from_prompt(prompt)
    responses: list[str] = []
    rewards: list[float] = []

    for tokens in grouped_tokens:
        response = tokenizer.decode(tokens)
        responses.append(response)
        reward_dict = keyword_inclusion_reward(response, keywords)
        rewards.append(reward_dict["reward"])

    return responses, rewards, keywords
