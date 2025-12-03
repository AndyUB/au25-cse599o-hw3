import matplotlib.pyplot as plt
import numpy as np

def plot_kl_rewards(
    ref_kls: list[float],
    old_kls: list[float],
    avg_train_rewards: list[float],
    avg_val_rewards: list[float],
    out_file: str,
) -> None:
    """
    Plot KL divergences and average rewards over training iterations.

    Args:
        ref_kls (list[float]): List of reference model KL divergences.
        old_kls (list[float]): List of old policy KL divergences.
        avg_train_rewards (list[float]): List of average training rewards.
        avg_val_rewards (list[float]): List of average validation rewards.
        out_file (str): Output file path to save the plot.
    """
    num_iters = len(ref_kls)
    if num_iters == 0:
        return

    iters = np.arange(1, num_iters + 1)

    # Plot 4 lines on the same graph
    # x-axis: iters
    # y-axis: ref_kls, old_kls, avg_train_rewards, avg_val_rewards
    plt.figure()
    plt.plot(iters, ref_kls, label="KL w.r.t. Reference Model")
    plt.plot(iters, old_kls, label="KL w.r.t. Last Step's Policy")
    plt.plot(iters, avg_train_rewards, label="Avg Training Reward")
    plt.plot(iters, avg_val_rewards, label="Avg Validation Reward")
    plt.xlabel("Training Iteration")
    plt.ylabel("KL Divergence or Reward")
    plt.title("KL Divergences and Average Rewards Over Training Iterations")
    plt.legend()
    plt.savefig(out_file)
    plt.close()
