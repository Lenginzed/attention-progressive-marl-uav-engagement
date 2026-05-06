import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_STAGES = [
    "2v2/ResearchHandcrafted/Stage1",
    "2v2/ResearchHandcrafted/Stage2",
    "2v2/ResearchHandcrafted/Stage3",
]


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


def latest_metrics_path(results_root, scenario_name, experiment_prefix):
    base_dir = Path(results_root) / scenario_name / "mappo"
    if not base_dir.exists():
        raise RuntimeError(f"Scenario results not found: {base_dir}")
    stage_tag = scenario_name.split("/")[-1].lower()
    experiment_dirs = [
        path for path in base_dir.iterdir()
        if path.is_dir() and path.name.startswith(experiment_prefix) and stage_tag in path.name.lower()
    ]
    if len(experiment_dirs) == 0:
        experiment_dirs = [path for path in base_dir.iterdir() if path.is_dir()]
    if len(experiment_dirs) == 0:
        raise RuntimeError(f"No experiment directories found in {base_dir}")
    experiment_dir = sorted(experiment_dirs, key=lambda path: path.stat().st_mtime)[-1]
    run_dir = latest_run_dir(experiment_dir)
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        raise RuntimeError(f"metrics.jsonl not found in {run_dir}")
    return metrics_path


def select_records(records, prefer_eval):
    if prefer_eval:
        chosen = [record for record in records if record.get("event") == "eval"]
        if chosen:
            return chosen, "eval"
    chosen = [record for record in records if record.get("event") == "train"]
    if not chosen:
        raise RuntimeError("No train or eval records found.")
    return chosen, "train"


def get_series(records, prefix, key):
    metric_key = f"{prefix}{key}"
    return np.asarray([float(record.get(metric_key, 0.0)) for record in records], dtype=float)


def _window_stats(series, window):
    if len(series) == 0:
        return 0.0, 0.0
    window = max(1, min(int(window), len(series)))
    tail = np.asarray(series[-window:], dtype=float)
    return float(tail.mean()), float(tail.std(ddof=0))


def build_stage_summary(records, stage_name, final_window):
    train_records = [record for record in records if record.get("event") == "train"]
    eval_records = [record for record in records if record.get("event") == "eval"]
    chosen_records, chosen_mode = select_records(records, prefer_eval=True)
    draw_series = get_series(chosen_records, "eval_" if chosen_mode == "eval" else "", "draw_rate")
    win_series = get_series(chosen_records, "eval_" if chosen_mode == "eval" else "", "win_rate")
    loss_series = get_series(chosen_records, "eval_" if chosen_mode == "eval" else "", "loss_rate")
    resolved_series = win_series + loss_series
    low_altitude_series = get_series(chosen_records, "eval_" if chosen_mode == "eval" else "", "low_altitude_rate")
    timeout_series = get_series(chosen_records, "eval_" if chosen_mode == "eval" else "", "timeout_rate")
    shield_series = get_series(chosen_records, "eval_" if chosen_mode == "eval" else "", "safety_shield_rate")
    reward_key = "eval_average_episode_rewards" if chosen_mode == "eval" else "average_episode_rewards"
    reward_series = get_series(chosen_records, "", reward_key)
    final_win_mean, final_win_std = _window_stats(win_series, final_window)
    final_loss_mean, final_loss_std = _window_stats(loss_series, final_window)
    final_draw_mean, final_draw_std = _window_stats(draw_series, final_window)
    final_reward_mean, final_reward_std = _window_stats(reward_series, final_window)
    final_shield_mean, final_shield_std = _window_stats(shield_series, final_window)

    summary = {
        "stage_name": stage_name,
        "source": chosen_mode,
        "num_train_records": len(train_records),
        "num_eval_records": len(eval_records),
        "initial_draw_rate": float(draw_series[0]) if len(draw_series) else 0.0,
        "final_draw_rate": float(draw_series[-1]) if len(draw_series) else 0.0,
        "initial_win_rate": float(win_series[0]) if len(win_series) else 0.0,
        "final_win_rate": float(win_series[-1]) if len(win_series) else 0.0,
        "initial_loss_rate": float(loss_series[0]) if len(loss_series) else 0.0,
        "final_loss_rate": float(loss_series[-1]) if len(loss_series) else 0.0,
        "initial_resolved_rate": float(resolved_series[0]) if len(resolved_series) else 0.0,
        "final_resolved_rate": float(resolved_series[-1]) if len(resolved_series) else 0.0,
        "draw_resolution_gain": float(draw_series[0] - draw_series[-1]) if len(draw_series) else 0.0,
        "win_gain": float(win_series[-1] - win_series[0]) if len(win_series) else 0.0,
        "loss_gain": float(loss_series[-1] - loss_series[0]) if len(loss_series) else 0.0,
        "final_low_altitude_rate": float(low_altitude_series[-1]) if len(low_altitude_series) else 0.0,
        "final_timeout_rate": float(timeout_series[-1]) if len(timeout_series) else 0.0,
        "final_safety_shield_rate": float(shield_series[-1]) if len(shield_series) else 0.0,
        "final_reward": float(reward_series[-1]) if len(reward_series) else 0.0,
        "final_window_size": int(max(1, min(final_window, len(chosen_records)))) if len(chosen_records) else 0,
        "final_window_win_rate_mean": final_win_mean,
        "final_window_win_rate_std": final_win_std,
        "final_window_loss_rate_mean": final_loss_mean,
        "final_window_loss_rate_std": final_loss_std,
        "final_window_draw_rate_mean": final_draw_mean,
        "final_window_draw_rate_std": final_draw_std,
        "final_window_reward_mean": final_reward_mean,
        "final_window_reward_std": final_reward_std,
        "final_window_safety_shield_rate_mean": final_shield_mean,
        "final_window_safety_shield_rate_std": final_shield_std,
        "best_win_rate": float(np.max(win_series)) if len(win_series) else 0.0,
        "best_resolved_rate": float(np.max(resolved_series)) if len(resolved_series) else 0.0,
    }
    return summary


def plot_stage_trends(stage_payloads, output_path, smooth_window):
    fig, axes = plt.subplots(len(stage_payloads), 1, figsize=(12, 4 * len(stage_payloads)), sharex=False)
    if len(stage_payloads) == 1:
        axes = [axes]

    for axis, payload in zip(axes, stage_payloads):
        records, mode = select_records(payload["records"], prefer_eval=True)
        prefix = "eval_" if mode == "eval" else ""
        steps = np.asarray([record["total_num_steps"] for record in records], dtype=float)
        draw_rate = moving_average(get_series(records, prefix, "draw_rate"), smooth_window)
        win_rate = moving_average(get_series(records, prefix, "win_rate"), smooth_window)
        loss_rate = moving_average(get_series(records, prefix, "loss_rate"), smooth_window)
        resolved_rate = moving_average(win_rate + loss_rate, smooth_window)

        axis.plot(steps, draw_rate, label="Draw Rate", linewidth=2)
        axis.plot(steps, win_rate, label="Win Rate", linewidth=2)
        axis.plot(steps, loss_rate, label="Loss Rate", linewidth=2)
        axis.plot(steps, resolved_rate, label="Resolved (Win+Loss)", linewidth=2, linestyle="--")
        axis.set_title(f"{payload['stage_name']} ({mode})")
        axis.set_ylabel("Rate")
        axis.set_ylim(0.0, 1.02)
        axis.grid(True, linestyle="--", alpha=0.4)
        axis.legend()

    axes[-1].set_xlabel("Environment Steps")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_stage_comparison(stage_payloads, output_path):
    stage_names = [payload["stage_name"].split("/")[-1] for payload in stage_payloads]
    summaries = [payload["summary"] for payload in stage_payloads]
    x = np.arange(len(stage_names))

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    win_vals = np.asarray([summary["final_win_rate"] for summary in summaries], dtype=float)
    loss_vals = np.asarray([summary["final_loss_rate"] for summary in summaries], dtype=float)
    draw_vals = np.asarray([summary["final_draw_rate"] for summary in summaries], dtype=float)
    axes[0].bar(x, win_vals, label="Win")
    axes[0].bar(x, loss_vals, bottom=win_vals, label="Loss")
    axes[0].bar(x, draw_vals, bottom=win_vals + loss_vals, label="Draw")
    axes[0].set_xticks(x, stage_names)
    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_ylabel("Final Outcome Rate")
    axes[0].set_title("Three-Stage Final Outcome Comparison")
    axes[0].grid(True, linestyle="--", alpha=0.4, axis="y")
    axes[0].legend()

    width = 0.24
    low_vals = np.asarray([summary["final_low_altitude_rate"] for summary in summaries], dtype=float)
    timeout_vals = np.asarray([summary["final_timeout_rate"] for summary in summaries], dtype=float)
    shield_vals = np.asarray([summary["final_safety_shield_rate"] for summary in summaries], dtype=float)
    axes[1].bar(x - width, low_vals, width=width, label="Low Altitude")
    axes[1].bar(x, timeout_vals, width=width, label="Timeout")
    axes[1].bar(x + width, shield_vals, width=width, label="Safety Shield")
    axes[1].set_xticks(x, stage_names)
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_ylabel("Rate")
    axes[1].set_title("Termination / Safety Comparison")
    axes[1].grid(True, linestyle="--", alpha=0.4, axis="y")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyze staged curriculum results and visualize draw-to-win/loss conversion.")
    parser.add_argument("--results-root", type=str, default=None, help="Root directory containing MultipleCombat results.")
    parser.add_argument("--stages", nargs="+", default=DEFAULT_STAGES, help="Scenario names to analyze.")
    parser.add_argument("--experiment-prefix", type=str, default="curriculum_handcrafted", help="Experiment prefix used in training.")
    parser.add_argument("--smooth-window", type=int, default=5, help="Moving-average window for trend figures.")
    parser.add_argument("--final-window", type=int, default=5, help="Number of trailing eval records used for final mean±std statistics.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to store plots and JSON summary.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    results_root = Path(args.results_root) if args.results_root else repo_root / "scripts" / "results" / "MultipleCombat"
    output_dir = Path(args.output_dir) if args.output_dir else repo_root / "scripts" / "results" / "analysis" / args.experiment_prefix
    output_dir.mkdir(parents=True, exist_ok=True)

    stage_payloads = []
    for stage_name in args.stages:
        metrics_path = latest_metrics_path(results_root, stage_name, args.experiment_prefix)
        records = load_records(metrics_path)
        summary = build_stage_summary(records, stage_name, args.final_window)
        stage_payloads.append({
            "stage_name": stage_name,
            "metrics_path": str(metrics_path),
            "records": records,
            "summary": summary,
        })

    plot_stage_trends(stage_payloads, output_dir / "draw_resolution_trends.png", args.smooth_window)
    plot_stage_comparison(stage_payloads, output_dir / "stage_comparison.png")

    summary_path = output_dir / "curriculum_summary.json"
    with summary_path.open("w", encoding="utf-8") as file_obj:
        json.dump(
            {
                "results_root": str(results_root),
                "experiment_prefix": args.experiment_prefix,
                "stages": [
                    {
                        "stage_name": payload["stage_name"],
                        "metrics_path": payload["metrics_path"],
                        "summary": payload["summary"],
                    }
                    for payload in stage_payloads
                ],
            },
            file_obj,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved trend figure to {output_dir / 'draw_resolution_trends.png'}")
    print(f"Saved comparison figure to {output_dir / 'stage_comparison.png'}")
    print(f"Saved summary JSON to {summary_path}")


if __name__ == "__main__":
    main()
