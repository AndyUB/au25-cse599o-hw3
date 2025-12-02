import math
import torch


def cross_entropy_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
) -> torch.Tensor:
    """Compute the cross-entropy loss between logits and target token IDs.

    Args:
        logits (torch.Tensor): Logits tensor of shape
            (..., seq_len, vocab_size).
        target_ids (torch.Tensor): Target token IDs of shape (..., seq_len).

    Returns:
        torch.Tensor: Scalar tensor representing the average cross-entropy
            loss over the batch.
    """
    if target_ids.dtype != torch.long:
        target_ids = target_ids.long()

    max_logits = torch.max(logits, dim=-1, keepdim=True).values
    shifted = logits - max_logits
    exp = torch.exp(shifted)
    sum_exp = torch.sum(exp, dim=-1, keepdim=True)
    unshifted = max_logits + torch.log(sum_exp)
    target_logits = torch.gather(logits, dim=-1, index=target_ids.unsqueeze(-1))
    neg_log_likelihood = (unshifted - target_logits).squeeze(-1)
    return torch.mean(neg_log_likelihood)


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float,
        betas: tuple[float, float],
        eps: float,
        weight_decay: float,
        dtype: torch.dtype | None = None,
    ):
        """Initialize the AdamW optimizer.

        Args:
            params: Parameters to optimize.
            lr (float): Learning rate.
            betas ((float, float)): beta1 and beta2 coefficients.
            eps (float): Small constant for numerical stability.
            weight_decay (float): Weight decay coefficient.
            dtype (torch.dtype | None): Data type for the optimizer states.
        """

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "dtype": dtype,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            dtype = group["dtype"]

            for p in group["params"]:
                p: torch.Tensor
                g: torch.Tensor = p.grad
                if g is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p, dtype=dtype)
                    state["v"] = torch.zeros_like(p, dtype=dtype)
                m: torch.Tensor = state["m"]
                v: torch.Tensor = state["v"]
                state["t"] += 1
                t = state["t"]

                m.mul_(beta1).add_((1 - beta1) * g)
                v.mul_(beta2).add_((1 - beta2) * g * g)
                alpha_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                p.sub_(alpha_t * m / (torch.sqrt(v) + eps))
                if weight_decay != 0:
                    p.sub_(lr * weight_decay * p)

        return loss
