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
        raise ValueError("Prompt format is incorrect")

    kw_str = prompt[
        len(KEYWORD_INCLUSION_PROMPT_PREFIX) : -len(KEYWORD_INCLUSION_PROMPT_SUFFIX)
    ]
    keywords = kw_str.split(", ")
    return keywords


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
def generate_response(
    model: torch.nn.Module,
    tokenizer: tiktoken.Encoding,
    prompt: str,
    max_tokens: int,
    temperature: float,
    context_length: int,
    device: torch.device,
) -> tuple[list[int], torch.Tensor]:
    """
    Generate a response from the model given a prompt.

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
