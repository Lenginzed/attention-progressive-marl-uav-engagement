import argparse
import json
from pathlib import Path

import numpy as np


BASELINES = {
    "ippo": {
        "label": "Independent PPO",
        "algorithm_name": "ippo",
        "variant_name": "ippo",
    },
    "ippo_structured": {
        "label": "Structured IPPO",
        "algorithm_name": "ippo",
        "variant_name": "ippo_structured",
    },
    "matd3": {
        "label": "MATD3",
        "algorithm_name": "matd3",
        "variant_name": "matd3",
    },
}


def latest_run_dir(base_dir):
    candidates = [path for path in base_dir.iterdir() if path.is_dir() and path.name.startswith("run")]
    if not candidates:
        raise RuntimeError(f"No run directories found in {base_dir}")
    return sorted(candidates, key=lambda path: int(path.name[3:]) if path.name[3:].isdigit() else -1)[-1]


def latest_metrics_path(results_root, scenario_name, algorithm_name, experiment_name):
    base_dir = Path(results_root) / "MultipleCombat" / scenario_name / algorithm_name / experiment_name
    if not base_dir.exists():
        raise RuntimeError(f"Experiment directory not found: {base_dir}")
    run_dir = latest_run_dir(base_dir)
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        raise RuntimeError(f"metrics.jsonl not found in {run_dir}")
    return metrics_path


def load_records(path):
    records = []
    with open(path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def select_records(records):
    eval_records = [record for record in records if record.get("event") == "eval"]
    if eval_records:
        return eval_records, "eval_"
    train_records = [record for record in records if record.get("event") == "train"]
    if train_records:
        return train_records, ""
    raise RuntimeError("No train or eval records found.")


def series(records, prefix, key):
    return np.asarray([float(record.get(f"{prefix}{key}", 0.0)) for record in records], dtype=float)


def summarize_run(records):
    chosen, prefix = select_records(records)
    win = series(chosen, prefix, "win_rate")
    loss = series(chosen, prefix, "loss_rate")
    draw = series(chosen, prefix, "draw_rate")
    reward_key = "eval_average_episode_rewards" if prefix == "eval_" else "average_episode_rewards"
    reward = np.asarray([float(record.get(reward_key, 0.0)) for record in chosen], dtype=float)
    return {
        "final_win_rate": float(win[-1]) if len(win) else 0.0,
        "final_loss_rate": float(loss[-1]) if len(loss) else 0.0,
        "final_draw_rate": float(draw[-1]) if len(draw) else 0.0,
        "final_reward": float(reward[-1]) if len(reward) else 0.0,
        "best_win_rate": float(np.max(win)) if len(win) else 0.0,
    }


def aggregate(runs):
    metrics = ["final_win_rate", "final_loss_rate", "final_draw_rate", "final_reward", "best_win_rate"]
    payload = {}
    for metric in metrics:
        values = np.asarray([run[metric] for run in runs], dtype=float)
        payload[f"{metric}_mean"] = float(values.mean()) if len(values) else 0.0
        payload[f"{metric}_std"] = float(values.std(ddof=0)) if len(values) else 0.0
    return payload


def main():
    parser = argparse.ArgumentParser(description="Analyze external baseline runs and export a benchmark-ready summary.")
    parser.add_argument("--scenario-name", type=str, default="2v2/ResearchHandcrafted/Stage2")
    parser.add_argument("--experiment-prefix", type=str, default="external_s1s2")
    parser.add_argument("--baselines", nargs="+", default=["ippo"], choices=list(BASELINES.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--results-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    results_root = Path(args.results_root) if args.results_root else repo_root / "scripts" / "results"
    output_dir = Path(args.output_dir) if args.output_dir else repo_root / "scripts" / "results" / "analysis" / args.experiment_prefix
    output_dir.mkdir(parents=True, exist_ok=True)

    baselines_payload = []
    for baseline_name in args.baselines:
        baseline = BASELINES[baseline_name]
        runs = []
        for seed in args.seeds:
            experiment_name = f"{args.experiment_prefix}_{baseline_name}_seed{seed}"
            metrics_path = latest_metrics_path(results_root, args.scenario_name, baseline["algorithm_name"], experiment_name)
            summary = summarize_run(load_records(metrics_path))
            runs.append({
                "seed": seed,
                "experiment_name": experiment_name,
                "metrics_path": str(metrics_path),
                "model_dir": str(Path(metrics_path).parent),
                "summary": summary,
            })
        best_run = max(
            runs,
            key=lambda run: (
                run["summary"]["final_win_rate"],
                run["summary"]["best_win_rate"],
                run["summary"]["final_reward"],
            ),
        )
        baselines_payload.append({
            "name": baseline_name,
            "label": baseline["label"],
            "variant_name": baseline["variant_name"],
            "algorithm_name": baseline["algorithm_name"],
            "scenario_name": args.scenario_name,
            "aggregate": aggregate([run["summary"] for run in runs]),
            "seed": best_run["seed"],
            "experiment_name": best_run["experiment_name"],
            "model_dir": best_run["model_dir"],
            "runs": runs,
        })

    summary_path = output_dir / "external_baselines_summary.json"
    with summary_path.open("w", encoding="utf-8") as file_obj:
        json.dump(
            {
                "scenario_name": args.scenario_name,
                "experiment_prefix": args.experiment_prefix,
                "baselines": baselines_payload,
            },
            file_obj,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved external baseline summary to {summary_path}")


if __name__ == "__main__":
    main()
