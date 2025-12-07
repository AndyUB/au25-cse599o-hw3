from typing import Callable
import matplotlib.pyplot as plt
import numpy as np
import os
import json


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


def grep_stats(
    log_path: str,
    key_fn: Callable[[str], bool] | None = None,
    skip_line_fn: Callable[[str], bool] | None = None,
) -> list[dict[str, float]]:
    """
    Extract per-iteration statistics from a log file.

    Args:
        log_path (str): Path to the log file.
        key_fn (Callable[[str], bool] | None): Optional function to filter keys.
        skip_line_fn (Callable[[str], bool] | None): Optional function to skip lines.

    Returns:
        list[dict[str, float]]: List of dictionaries containing statistics
            per iteration.
    """
    # Format:
    # Step <step>: {<dict[str, float | stat_str]>}
    # stat_str := <float> (<pct>%)
    stats_list = []
    with open(log_path, "r") as f:
        for line in f:
            if not line.startswith("Step "):
                continue
            if skip_line_fn is not None and skip_line_fn(line):
                continue
            # Remove single quotes
            line = line.replace("'", "")
            # Extract the dictionary string
            left_brace = line.index("{")
            right_brace = line.index("}")
            stat_str = line[left_brace + 1 : right_brace]
            # Extract key-value pairs
            stat_items = stat_str.split(", ")
            stat_dict: dict[str, float] = {}
            for item in stat_items:
                key, val_str = item.split(": ")
                # Skip keys that do not satisfy key_fn
                if key_fn is not None and not key_fn(key):
                    continue
                if "(" in val_str:
                    val = float(val_str.split(" (")[0])
                else:
                    val = float(val_str)
                stat_dict[key] = val
            stats_list.append(stat_dict)
    return stats_list


COLOR_MAP = {
    "rollout_time_ms": "#1f77b4",
    "update_time_ms": "#ff7f0e",
    "transfer_time_ms": "#2ca02c",
    "reference_time_ms": "#d62728",
    "other_time_ms": "#9467bd",
}


def plot_time_breakdown(log_path: str) -> None:
    """
    Plot time breakdown from a log file.

    Args:
        log_path (str): Path to the log file containing time breakdown data.
    """
    time_key_fn = lambda key: "time_ms" in key
    time_stats_list = grep_stats(log_path, key_fn=time_key_fn)
    if not time_stats_list:
        return

    total_key = "total_time_ms"
    breakdown_keys = [
        "rollout_time_ms",
        "update_time_ms",
        "transfer_time_ms",
        "reference_time_ms",
    ]
    other_key = "other_time_ms"
    breakdown_stats = {}
    for key in breakdown_keys:
        times = [stats[key] for stats in time_stats_list]
        if all(t == 0 for t in times):
            continue
        breakdown_stats[key] = times
    total_times = [stats[total_key] for stats in time_stats_list]
    other_times = []
    for i in range(len(time_stats_list)):
        sum_time = sum(breakdown_stats[key][i] for key in breakdown_stats.keys())
        other_times.append(total_times[i] - sum_time)
    breakdown_stats[other_key] = other_times
    breakdown_keys = [key for key in breakdown_keys if key in breakdown_stats]
    breakdown_keys.append(other_key)
    total_times = np.array(total_times, dtype=float) / 1000.0
    for key in breakdown_stats:
        breakdown_stats[key] = [t / 1000.0 for t in breakdown_stats[key]]

    num_iters = len(time_stats_list)
    bar_width = 0.6

    fig, ax = plt.subplots(figsize=(max(6, num_iters * 0.4), 6))
    iters = np.arange(1, num_iters + 1)
    bottom = np.zeros(num_iters, dtype=float)
    for key in breakdown_keys:
        vals = np.array(breakdown_stats[key], dtype=float)
        ax.bar(
            iters,
            vals,
            bar_width,
            bottom=bottom,
            label=key,
            align="center",
            edgecolor="black",
            linewidth=0.3,
            color=COLOR_MAP[key],
        )
        bottom += vals

    for x, total in zip(iters, total_times):
        ax.text(
            x,
            total,
            f"{total:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )

    ax.set_xticks(iters)
    ax.set_xticklabels([str(i) for i in iters], rotation=0)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Time (s)")
    ax.set_title("Time Breakdown per Iteration")
    ax.margins(x=0.05)

    key_to_name: Callable[[str], str] = (
        lambda s: s.replace("_time_ms", "").replace("_", " ").title()
    )
    handles, _ = ax.get_legend_handles_labels()
    labels = [key_to_name(k) for k in breakdown_keys]
    ax.legend(handles, labels, loc="upper right", fontsize=8)

    fig.tight_layout()

    out_path_bar = log_path + ".time_breakdown.png"
    fig.savefig(out_path_bar, dpi=150)
    plt.close(fig)

    breakdown_sums = np.array([sum(breakdown_stats[k]) for k in breakdown_keys])
    colors = [COLOR_MAP[k] for k in breakdown_keys]
    plot_time_breakdown_pie(
        breakdown_sums,
        labels,
        colors,
        "Time Breakdown (All Iterations)",
        log_path + ".time_breakdown_pie_all.png",
    )

    # All iterations except first (skip warmup)
    if num_iters > 1:
        breakdown_first = np.array([breakdown_stats[k][0] for k in breakdown_keys])
        breakdown_sums_wo_first = breakdown_sums - breakdown_first
        plot_time_breakdown_pie(
            breakdown_sums_wo_first,
            labels,
            colors,
            "Time Breakdown (All Iterations Except First)",
            log_path + ".time_breakdown_pie.png",
        )


def plot_time_breakdown_pie(
    breakdown: np.ndarray,
    labels: list[str],
    colors: list[str],
    title: str,
    out_file: str,
) -> None:
    """
    Plot a pie chart for time breakdown.

    Args:
        breakdown (np.ndarray): Array of time breakdown values.
        labels (list[str]): List of labels for each segment.
        colors (list[str]): List of colors for each segment.
        title (str): Title of the pie chart.
        out_file (str): Output file path to save the plot.
    """
    total = breakdown.sum()
    percents = breakdown / total * 100.0
    mask = percents >= 0.01
    breakdown = breakdown[mask]
    percents = percents[mask]
    labels = [lab for lab, keep in zip(labels, mask) if keep]
    colors = [col for col, keep in zip(colors, mask) if keep]

    legend_labels = [f"{lab} ({p:.2f}%)" for lab, p in zip(labels, percents)]
    pct_eps = 3
    explode = [0.08 if p > 0 and p < pct_eps else 0.0 for p in percents]

    fig, ax = plt.subplots(figsize=(6, 4))
    wedges, _ = ax.pie(
        breakdown,
        labels=None,
        colors=colors,
        startangle=90,
        explode=explode,
    )
    ax.axis("equal")
    ax.set_title(title)

    # Legend instead of on-slice labels
    ax.legend(
        wedges,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def plot_time_series(log_dir: str) -> None:
    """
    Plot a time series from log files in the specified directory.

    Args:
        log_dir (str): Directory containing log files.
    """
    if not os.path.isdir(log_dir):
        raise ValueError(f"{log_dir} is not a valid directory")

    time_keys = ["rollout_time_ms", "update_time_ms", "transfer_time_ms"]
    time_key_fn = lambda key: key in time_keys
    skip_warmup_fn = lambda line: "warmup=True" in line
    time_key_to_name = {
        "rollout_time_ms": "Rollout Generation",
        "update_time_ms": "Policy Update",
        "transfer_time_ms": "Weight Synchronization",
    }

    # k -> list of (time_key, time_start, time_end)
    k_to_time_series: dict[int, list[tuple[str, float, float]]] = {}
    # get log files ending with .log
    log_files = [f for f in os.listdir(log_dir) if f.endswith(".log")]
    hyperparams_found = False
    prompts_per_step: int | None = None
    num_steps: int | None = None
    for log_file in log_files:
        if "-k" not in log_file:
            continue
        k_str = log_file.split("-k")[-1].split(".log")[0]
        try:
            k = int(k_str)
        except ValueError:
            continue

        log_path = os.path.join(log_dir, log_file)
        stats_list = grep_stats(
            log_path, key_fn=time_key_fn, skip_line_fn=skip_warmup_fn
        )

        if not hyperparams_found:
            with open(log_path, "r") as f:
                # first line contains hyperparameters
                first_line = f.readline()
                left_paren = first_line.index("(")
                right_paren = first_line.index(")")
                hyperparam_str = first_line[left_paren + 1 : right_paren]
                hyperparam_items = hyperparam_str.split(", ")
                for item in hyperparam_items:
                    hyperparam_key, hyperparam_val = item.split("=")
                    if hyperparam_key == "prompts_per_batch":
                        prompts_per_step = int(hyperparam_val)
                    elif hyperparam_key == "steps":
                        num_steps = int(hyperparam_val)
                if prompts_per_step is not None and num_steps is not None:
                    hyperparams_found = True

        time_series = []
        global_time = 0.0
        for stats in stats_list:
            for time_key in time_keys:
                time_val = stats.get(time_key, 0.0) / 1000.0  # convert to seconds
                time_start = global_time
                time_end = global_time + time_val
                time_series.append((time_key, time_start, time_end))
                global_time = time_end
        k_to_time_series[k] = time_series

    # Plot time series for each k on the same plot
    # x-axis: time (s)
    # y-axis: for each k, a horizontal bar with segments
    # Each segment colored by time_key, starting from time_start to time_end
    fig, ax = plt.subplots(figsize=(14, 3))
    y_ticks = []
    y_tick_labels = []
    for i, (k, time_series) in enumerate(sorted(k_to_time_series.items())):
        y = 0.25 * i
        y_ticks.append(y)
        y_tick_labels.append(f"k={k}")
        for time_key, time_start, time_end in time_series:
            ax.barh(
                y,
                time_end - time_start,
                left=time_start,
                height=0.2,
                label=time_key,
                align="center",
                color=COLOR_MAP[time_key],
            )

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels)
    ax.set_xlabel("Time (s)")
    title = "Training Timelines"
    if hyperparams_found:
        title += f" ({num_steps} steps, {prompts_per_step} prompts/step)"
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    uniqs = {}
    for h, l in zip(handles, labels):
        l = time_key_to_name[l]
        if l not in uniqs:
            uniqs[l] = h
    ax.legend(uniqs.values(), uniqs.keys(), loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(log_dir, "time_series.png"), dpi=150)
    plt.close(fig)

    k_to_total_time = {
        k: time_series[-1][2] for k, time_series in k_to_time_series.items()
    }
    sync_total_time = k_to_total_time[1]
    k_to_speedup_stats = {}
    for k, total_time in k_to_total_time.items():
        speedup = sync_total_time / total_time
        speedup_stats = {
            "total_time_s": total_time,
            "speedup_ratio": speedup,
        }
        if hyperparams_found:
            # Samples per second, 4 samples per prompt
            throughput = (prompts_per_step * num_steps * 4) / total_time
            speedup_stats["throughput_samples_per_s"] = throughput
        k_to_speedup_stats[k] = speedup_stats

    stats_path = os.path.join(log_dir, "speedup_stats.json")
    with open(stats_path, "w") as f:
        json.dump(k_to_speedup_stats, f, indent=4)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--mode", type=str, required=True, choices=["time_breakdown", "time_series"]
    )
    parser.add_argument("--log_path", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    args = parser.parse_args()

    if args.mode == "time_breakdown":
        if args.log_path is None:
            raise ValueError("log_path must be provided for time_breakdown mode")
        plot_time_breakdown(args.log_path)
    elif args.mode == "time_series":
        if args.log_dir is None:
            raise ValueError("log_dir must be provided for time_series mode")
        plot_time_series(args.log_dir)
