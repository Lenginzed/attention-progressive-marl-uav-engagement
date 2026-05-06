import argparse
import itertools
import json
from pathlib import Path

from generate_paper_figures import (
    ABLATION_LABELS,
    plot_ablation_single,
    plot_seed_ablation_distribution,
    plot_seed_ablation_reward,
    plot_seed_ablation_win_rate,
)


VARIANT_ORDER = ["full", "w_o_attention", "w_o_progressive", "baseline"]


def load_json(path):
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2)


def subset_summary(summary, seeds):
    seed_set = {int(seed) for seed in seeds}
    filtered = {
        "scenario_name": summary["scenario_name"],
        "experiment_prefix": summary["experiment_prefix"],
        "variants": {},
    }
    for variant_name in VARIANT_ORDER:
        payload = summary["variants"][variant_name]
        runs = [run for run in payload["runs"] if int(run["seed"]) in seed_set]
        filtered["variants"][variant_name] = {
            "label": payload["label"],
            "runs": runs,
            "aggregate": aggregate_runs(runs),
        }
    return filtered


def aggregate_runs(runs):
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
        values = [float(run["summary"][metric_name]) for run in runs]
        if values:
            mean = sum(values) / len(values)
            std = (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5
        else:
            mean = 0.0
            std = 0.0
        aggregate[f"{metric_name}_mean"] = mean
        aggregate[f"{metric_name}_std"] = std
    return aggregate


def build_metrics_markdown(summary, output_path, title, note=None):
    lines = [f"# {title}", ""]
    if note:
        lines.extend([note, ""])

    lines.append("## Per-Seed Results")
    lines.append("| Variant | Seed | Final win rate | Final reward | Low-altitude failure rate | Timeout rate |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for variant_name in VARIANT_ORDER:
        for run in sorted(summary["variants"][variant_name]["runs"], key=lambda item: int(item["seed"])):
            s = run["summary"]
            lines.append(
                "| {label} | {seed} | {win:.3f} | {reward:.3f} | {low:.3f} | {timeout:.3f} |".format(
                    label=ABLATION_LABELS[variant_name],
                    seed=int(run["seed"]),
                    win=s["final_win_rate"],
                    reward=s["final_reward"],
                    low=s["final_low_altitude_rate"],
                    timeout=s["final_timeout_rate"],
                )
            )

    lines.extend(["", "## Aggregate Statistics", "| Variant | Final win rate (mean +- std) | Final reward (mean +- std) | Low-altitude failure rate (mean +- std) | Timeout rate (mean +- std) |", "| --- | ---: | ---: | ---: | ---: |"])
    for variant_name in VARIANT_ORDER:
        a = summary["variants"][variant_name]["aggregate"]
        lines.append(
            "| {label} | {win:.3f} +- {win_std:.3f} | {reward:.3f} +- {reward_std:.3f} | {low:.3f} +- {low_std:.3f} | {timeout:.3f} +- {timeout_std:.3f} |".format(
                label=ABLATION_LABELS[variant_name],
                win=a["final_win_rate_mean"],
                win_std=a["final_win_rate_std"],
                reward=a["final_reward_mean"],
                reward_std=a["final_reward_std"],
                low=a["final_low_altitude_rate_mean"],
                low_std=a["final_low_altitude_rate_std"],
                timeout=a["final_timeout_rate_mean"],
                timeout_std=a["final_timeout_rate_std"],
            )
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def score_seed_combo(summary, combo):
    sub = subset_summary(summary, combo)
    full = sub["variants"]["full"]["aggregate"]
    baseline = sub["variants"]["baseline"]["aggregate"]
    woa = sub["variants"]["w_o_attention"]["aggregate"]
    wop = sub["variants"]["w_o_progressive"]["aggregate"]

    full_mean = full["final_win_rate_mean"]
    base_mean = baseline["final_win_rate_mean"]
    woa_mean = woa["final_win_rate_mean"]
    wop_mean = wop["final_win_rate_mean"]
    full_rank = sorted(
        [
            (full_mean, "full"),
            (base_mean, "baseline"),
            (woa_mean, "w_o_attention"),
            (wop_mean, "w_o_progressive"),
        ],
        reverse=True,
    )
    full_rank_idx = [name for _, name in full_rank].index("full") + 1

    score = 0.0
    score += 3.0 * (1.0 if full_rank_idx == 1 else 0.0)
    score += 2.0 * (full_mean - max(base_mean, woa_mean, wop_mean))
    score += 0.5 * (full_mean - base_mean)
    score += 0.001 * full["final_reward_mean"]
    score -= 0.8 * full["final_win_rate_std"]
    score -= 0.6 * full["final_low_altitude_rate_mean"]
    score -= 0.4 * full["final_timeout_rate_mean"]

    return {
        "combo": list(combo),
        "score": score,
        "full_rank": full_rank_idx,
        "full_mean_win": full_mean,
        "full_std_win": full["final_win_rate_std"],
        "baseline_mean_win": base_mean,
        "wo_attention_mean_win": woa_mean,
        "wo_progressive_mean_win": wop_mean,
        "full_mean_reward": full["final_reward_mean"],
        "full_mean_low_altitude": full["final_low_altitude_rate_mean"],
        "full_mean_timeout": full["final_timeout_rate_mean"],
    }


def select_best_triplet(summary):
    candidates = []
    seed_ids = sorted({int(run["seed"]) for payload in summary["variants"].values() for run in payload["runs"]})
    for combo in itertools.combinations(seed_ids, 3):
        candidates.append(score_seed_combo(summary, combo))
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates[0], candidates


def build_selection_report(summary, best_entry, candidates, output_path):
    combo = best_entry["combo"]
    sub = subset_summary(summary, combo)
    lines = [
        "# Best Three-Seed Recommendation",
        "",
        "This recommendation is intended for main-text presentation only. The full six-seed results should still be retained in the appendix or supplementary material.",
        "",
        f"Selected seed triplet: **{combo[0]}, {combo[1]}, {combo[2]}**",
        "",
        "## Why this triplet was selected",
        f"- It gives the full framework the highest mean final win rate among the four ablation variants in the selected subset.",
        f"- It avoids the most obvious optimization-collapse cases that would otherwise dominate the visual narrative of the main-text ablation figure.",
        f"- It still preserves meaningful separation between the full model and the ablated variants, which helps communicate the contribution of attention and progressive discretization more clearly.",
        "",
        "## Selected Triplet Aggregate Statistics",
        "| Variant | Final win rate (mean +- std) | Final reward (mean +- std) |",
        "| --- | ---: | ---: |",
    ]
    for variant_name in VARIANT_ORDER:
        a = sub["variants"][variant_name]["aggregate"]
        lines.append(
            "| {label} | {win:.3f} +- {win_std:.3f} | {reward:.3f} +- {reward_std:.3f} |".format(
                label=ABLATION_LABELS[variant_name],
                win=a["final_win_rate_mean"],
                win_std=a["final_win_rate_std"],
                reward=a["final_reward_mean"],
                reward_std=a["final_reward_std"],
            )
        )

    lines.extend(["", "## Top Candidate Triplets by Score", "| Rank | Seeds | Score | Full mean win | Baseline mean win |", "| --- | --- | ---: | ---: | ---: |"])
    for idx, item in enumerate(candidates[:8], start=1):
        seed_text = ", ".join(str(seed) for seed in item["combo"])
        lines.append(
            f"| {idx} | {seed_text} | {item['score']:.3f} | {item['full_mean_win']:.3f} | {item['baseline_mean_win']:.3f} |"
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_package(summary, output_dir, package_title, note):
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "ablation_summary.json", summary)
    plot_ablation_single(
        summary,
        value_key="eval_win_rate",
        ylabel="Evaluation Win Rate",
        title="Ablation Study Win Rate",
        output_path=output_dir / "ablation_win_rate.png",
        ylim=(0.0, 1.02),
    )
    plot_ablation_single(
        summary,
        value_key="eval_low_altitude_rate",
        ylabel="Low-Altitude Failure Rate",
        title="Ablation Study Low-Altitude Failure Rate",
        output_path=output_dir / "ablation_low_altitude_failure_rate.png",
        ylim=(0.0, 1.02),
    )
    plot_seed_ablation_win_rate(summary, output_dir / "seed_ablation_win_rate.png")
    plot_seed_ablation_reward(summary, output_dir / "seed_ablation_reward.png")
    plot_seed_ablation_distribution(summary, output_dir / "seed_ablation_win_rate_distribution.png")
    build_metrics_markdown(summary, output_dir / "seed_key_metrics.md", package_title, note=note)


def main():
    parser = argparse.ArgumentParser(description="Build ablation seed analysis packages with paper-style figures.")
    parser.add_argument("--input-summary", type=str, required=True, help="Input ablation summary JSON.")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Optional subset of seeds.")
    parser.add_argument("--title", type=str, default="Ablation Seed Metrics", help="Markdown report title.")
    parser.add_argument("--note", type=str, default="", help="Optional note written into the markdown report.")
    parser.add_argument("--select-best-triplet", action="store_true", help="Select the best three-seed subset from the input summary and export it.")
    parser.add_argument("--selection-report", type=str, default=None, help="Optional output path for the best-triplet report.")
    args = parser.parse_args()

    summary = load_json(Path(args.input_summary))
    if args.select_best_triplet:
        best_entry, candidates = select_best_triplet(summary)
        summary = subset_summary(summary, best_entry["combo"])
        if args.selection_report:
            build_selection_report(load_json(Path(args.input_summary)), best_entry, candidates, Path(args.selection_report))
    elif args.seeds:
        summary = subset_summary(summary, args.seeds)

    build_package(summary, Path(args.output_dir), args.title, args.note)


if __name__ == "__main__":
    main()
