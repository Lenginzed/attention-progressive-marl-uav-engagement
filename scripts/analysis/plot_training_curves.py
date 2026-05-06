import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_records(path):
    records = []
    with open(path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def moving_average(values, window):
    if window <= 1 or len(values) == 0:
        return np.asarray(values, dtype=float)
    kernel = np.ones(window, dtype=float) / window
    padded = np.pad(np.asarray(values, dtype=float), (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to metrics.jsonl")
    parser.add_argument("--output", type=str, default=None, help="Output figure path")
    parser.add_argument("--smooth-window", type=int, default=5, help="Moving-average window")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name("training_curves.png")
    records = load_records(input_path)
    train_records = [record for record in records if record.get("event") == "train"]
    eval_records = [record for record in records if record.get("event") == "eval"]

    if len(train_records) == 0:
        raise RuntimeError(f"No train records found in {input_path}")

    train_steps = [record["total_num_steps"] for record in train_records]
    train_rewards = moving_average([record.get("average_episode_rewards", 0.0) for record in train_records], args.smooth_window)
    train_win_rates = moving_average([record.get("win_rate", 0.0) for record in train_records], args.smooth_window)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(train_steps, train_rewards, label="Train Reward", linewidth=2)
    if eval_records:
        axes[0].plot(
            [record["total_num_steps"] for record in eval_records],
            [record.get("eval_average_episode_rewards", 0.0) for record in eval_records],
            label="Eval Reward",
            linewidth=2,
        )
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Reward Curve")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()

    axes[1].plot(train_steps, train_win_rates, label="Train Win Rate", linewidth=2)
    if eval_records:
        axes[1].plot(
            [record["total_num_steps"] for record in eval_records],
            [record.get("eval_win_rate", 0.0) for record in eval_records],
            label="Eval Win Rate",
            linewidth=2,
        )
    axes[1].set_xlabel("Environment Steps")
    axes[1].set_ylabel("Win Rate")
    axes[1].set_title("Win-Rate Curve")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
