import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ABLATION_VARIANTS = {
    "full": "MAPPO + Attention + Progressive",
    "w_o_attention": "MAPPO + Progressive",
    "w_o_progressive": "MAPPO + Attention",
    "baseline": "Vanilla MAPPO",
}


def load_records(path):
    records = []
    with open(path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def moving_average(values, window):
    values = np.asarray(values, dtype=float)
    if len(values) == 0 or window <= 1:
        return values
    kernel = np.ones(window, dtype=float) / window
    padded = np.pad(values, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def latest_run_dir(base_dir):
    candidates = [path for path in base_dir.iterdir() if path.is_dir() and path.name.startswith("run")]
    if len(candidates) == 0:
        raise RuntimeError(f"No run directories found in {base_dir}")
    return sorted(candidates, key=lambda path: int(path.name[3:]) if path.name[3:].isdigit() else -1)[-1]


def latest_metrics_path(results_root, scenario_name, experiment_name):
    base_dir = Path(results_root) / "MultipleCombat" / scenario_name / "mappo" / experiment_name
    if not base_dir.exists():
        raise RuntimeError(f"Experiment directory not found: {base_dir}")
    run_dir = latest_run_dir(base_dir)
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        raise RuntimeError(f"metrics.jsonl not found in {run_dir}")
    return metrics_path


def select_records(records):
    eval_records = [record for record in records if record.get("event") == "eval"]
    if eval_records:
        return eval_records, "eval_"
    train_records = [record for record in records if record.get("event") == "train"]
    if train_records:
        return train_records, ""
    raise RuntimeError("No train or eval records found.")


def get_series(records, prefix, key):
    return np.asarray([float(record.get(f"{prefix}{key}", 0.0)) for record in records], dtype=float)


def build_run_summary(records):
    selected, prefix = select_records(records)
    steps = np.asarray([record["total_num_steps"] for record in selected], dtype=float)
    win = get_series(selected, prefix, "win_rate")
    loss = get_series(selected, prefix, "loss_rate")
    draw = get_series(selected, prefix, "draw_rate")
    low = get_series(selected, prefix, "low_altitude_rate")
    timeout = get_series(selected, prefix, "timeout_rate")
    shield = get_series(selected, prefix, "safety_shield_rate")
    reward_key = "average_episode_rewards"
    if prefix == "eval_":
        reward_key = "eval_average_episode_rewards"
    reward = np.asarray([float(record.get(reward_key, 0.0)) for record in selected], dtype=float)
    resolved = win + loss

    return {
        "steps": steps,
        "win": win,
        "loss": loss,
        "draw": draw,
        "resolved": resolved,
        "low_altitude": low,
        "timeout": timeout,
        "shield": shield,
        "reward": reward,
        "final_win_rate": float(win[-1]) if len(win) else 0.0,
        "final_loss_rate": float(loss[-1]) if len(loss) else 0.0,
        "final_draw_rate": float(draw[-1]) if len(draw) else 0.0,
        "final_resolved_rate": float(resolved[-1]) if len(resolved) else 0.0,
        "final_low_altitude_rate": float(low[-1]) if len(low) else 0.0,
        "final_timeout_rate": float(timeout[-1]) if len(timeout) else 0.0,
        "final_safety_shield_rate": float(shield[-1]) if len(shield) else 0.0,
        "final_reward": float(reward[-1]) if len(reward) else 0.0,
        "best_win_rate": float(np.max(win)) if len(win) else 0.0,
        "best_resolved_rate": float(np.max(resolved)) if len(resolved) else 0.0,
    }


def aggregate_variant_runs(run_summaries):
    final_metric_names = [
        "final_win_rate",
        "final_loss_rate",
        "final_draw_rate",
        "final_resolved_rate",
        "final_low_altitude_rate",
        "final_timeout_rate",
        "final_safety_shield_rate",
        "final_reward",
        "best_win_rate",
        "best_resolved_rate",
    ]
    aggregate = {}
    for metric_name in final_metric_names:
        values = np.asarray([summary[metric_name] for summary in run_summaries], dtype=float)
        aggregate[f"{metric_name}_mean"] = float(values.mean()) if len(values) else 0.0
        aggregate[f"{metric_name}_std"] = float(values.std(ddof=0)) if len(values) else 0.0
    return aggregate


def plot_metric_curves(variant_payloads, output_path, smooth_window):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    metric_layout = [
        ("win", "Win Rate"),
        ("draw", "Draw Rate"),
        ("resolved", "Resolved Rate (Win+Loss)"),
        ("low_altitude", "Low-Altitude Rate"),
    ]

    for axis, (metric_key, title) in zip(axes.flatten(), metric_layout):
        for variant_name, payload in variant_payloads.items():
            summaries = payload["run_summaries"]
            min_len = min(len(summary["steps"]) for summary in summaries)
            if min_len == 0:
                continue
            steps = summaries[0]["steps"][:min_len]
            stacked = np.stack([moving_average(summary[metric_key][:min_len], smooth_window) for summary in summaries], axis=0)
            mean = stacked.mean(axis=0)
            std = stacked.std(axis=0)
            axis.plot(steps, mean, linewidth=2, label=ABLATION_VARIANTS[variant_name])
            axis.fill_between(steps, np.clip(mean - std, 0.0, 1.0), np.clip(mean + std, 0.0, 1.0), alpha=0.15)
        axis.set_title(title)
        axis.set_ylabel("Rate")
        axis.set_ylim(0.0, 1.02)
        axis.grid(True, linestyle="--", alpha=0.4)

    axes[1, 0].set_xlabel("Environment Steps")
    axes[1, 1].set_xlabel("Environment Steps")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_variant_bar_chart(variant_payloads, output_path):
    variant_names = list(variant_payloads.keys())
    labels = [ABLATION_VARIANTS[name] for name in variant_names]
    x = np.arange(len(variant_names))
    width = 0.18

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    win = np.asarray([variant_payloads[name]["aggregate"]["final_win_rate_mean"] for name in variant_names], dtype=float)
    draw = np.asarray([variant_payloads[name]["aggregate"]["final_draw_rate_mean"] for name in variant_names], dtype=float)
    resolved = np.asarray([variant_payloads[name]["aggregate"]["final_resolved_rate_mean"] for name in variant_names], dtype=float)
    win_std = np.asarray([variant_payloads[name]["aggregate"]["final_win_rate_std"] for name in variant_names], dtype=float)
    draw_std = np.asarray([variant_payloads[name]["aggregate"]["final_draw_rate_std"] for name in variant_names], dtype=float)
    resolved_std = np.asarray([variant_payloads[name]["aggregate"]["final_resolved_rate_std"] for name in variant_names], dtype=float)

    axes[0].bar(x - width, win, width=width, yerr=win_std, capsize=4, label="Final Win")
    axes[0].bar(x, draw, width=width, yerr=draw_std, capsize=4, label="Final Draw")
    axes[0].bar(x + width, resolved, width=width, yerr=resolved_std, capsize=4, label="Final Resolved")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_ylabel("Rate")
    axes[0].set_title("Ablation Final Outcome Comparison")
    axes[0].grid(True, linestyle="--", alpha=0.4, axis="y")
    axes[0].legend()

    low = np.asarray([variant_payloads[name]["aggregate"]["final_low_altitude_rate_mean"] for name in variant_names], dtype=float)
    timeout = np.asarray([variant_payloads[name]["aggregate"]["final_timeout_rate_mean"] for name in variant_names], dtype=float)
    shield = np.asarray([variant_payloads[name]["aggregate"]["final_safety_shield_rate_mean"] for name in variant_names], dtype=float)
    low_std = np.asarray([variant_payloads[name]["aggregate"]["final_low_altitude_rate_std"] for name in variant_names], dtype=float)
    timeout_std = np.asarray([variant_payloads[name]["aggregate"]["final_timeout_rate_std"] for name in variant_names], dtype=float)
    shield_std = np.asarray([variant_payloads[name]["aggregate"]["final_safety_shield_rate_std"] for name in variant_names], dtype=float)

    axes[1].bar(x - width, low, width=width, yerr=low_std, capsize=4, label="Low Altitude")
    axes[1].bar(x, timeout, width=width, yerr=timeout_std, capsize=4, label="Timeout")
    axes[1].bar(x + width, shield, width=width, yerr=shield_std, capsize=4, label="Safety Shield")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_ylabel("Rate")
    axes[1].set_title("Ablation Safety / Failure Comparison")
    axes[1].grid(True, linestyle="--", alpha=0.4, axis="y")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyze ablation results for attention and progressive discretization.")
    parser.add_argument("--scenario-name", type=str, default="2v2/ResearchHandcrafted/Stage2",
                        help="Controlled ablation finetune scenario name.")
    parser.add_argument("--experiment-prefix", type=str, default="ablation_s1s2",
                        help="Experiment prefix used by the ablation training script.")
    parser.add_argument("--variants", nargs="+", default=list(ABLATION_VARIANTS.keys()),
                        choices=list(ABLATION_VARIANTS.keys()), help="Variants to analyze.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3], help="Seeds to aggregate.")
    parser.add_argument("--results-root", type=str, default=None, help="Root directory containing training results.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory used to store plots and JSON summary.")
    parser.add_argument("--smooth-window", type=int, default=3, help="Moving-average window for mean curves.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    results_root = Path(args.results_root) if args.results_root else repo_root / "scripts" / "results"
    output_dir = Path(args.output_dir) if args.output_dir else repo_root / "scripts" / "results" / "analysis" / args.experiment_prefix
    output_dir.mkdir(parents=True, exist_ok=True)

    variant_payloads = {}
    for variant_name in args.variants:
        run_summaries = []
        run_payloads = []
        for seed in args.seeds:
            experiment_name = f"{args.experiment_prefix}_{variant_name}_seed{seed}"
            metrics_path = latest_metrics_path(results_root, args.scenario_name, experiment_name)
            records = load_records(metrics_path)
            summary = build_run_summary(records)
            run_summaries.append(summary)
            run_payloads.append({
                "seed": seed,
                "experiment_name": experiment_name,
                "metrics_path": str(metrics_path),
                "summary": {key: value for key, value in summary.items() if not isinstance(value, np.ndarray)},
            })
        variant_payloads[variant_name] = {
            "label": ABLATION_VARIANTS[variant_name],
            "run_summaries": run_summaries,
            "runs": run_payloads,
            "aggregate": aggregate_variant_runs(run_summaries),
        }

    plot_metric_curves(variant_payloads, output_dir / "ablation_learning_curves.png", args.smooth_window)
    plot_variant_bar_chart(variant_payloads, output_dir / "ablation_final_comparison.png")

    summary_path = output_dir / "ablation_summary.json"
    with summary_path.open("w", encoding="utf-8") as file_obj:
        json.dump(
            {
                "scenario_name": args.scenario_name,
                "experiment_prefix": args.experiment_prefix,
                "variants": {
                    variant_name: {
                        "label": payload["label"],
                        "runs": payload["runs"],
                        "aggregate": payload["aggregate"],
                    }
                    for variant_name, payload in variant_payloads.items()
                },
            },
            file_obj,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved learning curves to {output_dir / 'ablation_learning_curves.png'}")
    print(f"Saved final comparison figure to {output_dir / 'ablation_final_comparison.png'}")
    print(f"Saved summary JSON to {summary_path}")


if __name__ == "__main__":
    main()
