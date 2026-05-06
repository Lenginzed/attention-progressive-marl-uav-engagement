import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(r"F:\code\LAG-master")
RESULTS_ROOT = ROOT / "scripts" / "results"
ANALYSIS_ROOT = RESULTS_ROOT / "analysis"
OUTPUT_ROOT = RESULTS_ROOT / "result_analysis" / "06_figures"

CURRICULUM_SUMMARY_PATH = ANALYSIS_ROOT / "curriculum_handcrafted" / "curriculum_summary.json"
ABLATION_SUMMARY_PATH = ANALYSIS_ROOT / "ablation_s1s2" / "ablation_summary.json"
BENCHMARK_SUMMARY_PATH = ANALYSIS_ROOT / "benchmark_focus_vanilla_ippo1_hard100_pretty" / "benchmark_summary.json"
MATD3_BENCHMARK_SUMMARY_PATH = ANALYSIS_ROOT / "benchmark_matd3" / "benchmark_summary.json"
EXTERNAL_BASELINE_SUMMARY_PATH = ANALYSIS_ROOT / "external_s1s2" / "external_baselines_summary.json"

CURRICULUM_COLORS = {
    "Stage1": "#4c956c",
    "Stage2": "#2f6db3",
    "Stage3": "#c43d3d",
}

ABLATION_COLORS = {
    "full": "#c43d3d",
    "w_o_attention": "#2f6db3",
    "w_o_progressive": "#4c956c",
    "baseline": "#7f7f7f",
}

ABLATION_LABELS = {
    "full": "MAPPO + Attention + Progressive",
    "w_o_attention": "MAPPO + Progressive",
    "w_o_progressive": "MAPPO + Attention",
    "baseline": "Vanilla MAPPO",
}


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path):
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def merged_benchmark_summary(primary_summary, matd3_summary=None):
    matchups = []
    primary_map = {item["name"]: item for item in primary_summary.get("matchups", [])}
    matd3_map = {item["name"]: item for item in (matd3_summary or {}).get("matchups", [])}

    ordered = [
        ("vs_vanilla_mappo", "Vanilla MAPPO"),
        ("vs_ippo", "IPPO"),
        ("vs_matd3", "MATD3"),
        ("vs_hard_handcrafted", "Hard Handcrafted"),
    ]
    for name, short_label in ordered:
        if name in primary_map:
            item = primary_map[name]
        elif name in matd3_map:
            item = matd3_map[name]
        else:
            continue
        item = dict(item)
        item["short_label"] = short_label
        matchups.append(item)

    return {
        "matchups": matchups,
    }


def moving_average(values, window=5):
    values = np.asarray(values, dtype=float)
    if len(values) == 0 or window <= 1:
        return values
    pad_left = window - 1
    padded = np.pad(values, (pad_left, 0), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")


def select_eval_records(metrics_path):
    records = load_jsonl(metrics_path)
    eval_records = [record for record in records if record.get("event") == "eval"]
    return eval_records if eval_records else [record for record in records if record.get("event") == "train"]


def stage_short_name(stage_name):
    return stage_name.split("/")[-1]


def prettify_axes(ax, xlabel=None, ylabel=None, title=None):
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=13, pad=10)
    ax.grid(True, linestyle="--", alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def curriculum_stage_records(curriculum_summary):
    payload = []
    for stage in curriculum_summary["stages"]:
        short = stage_short_name(stage["stage_name"])
        records = select_eval_records(Path(stage["metrics_path"]))
        payload.append((short, records, stage["summary"]))
    return payload


def plot_curriculum_single(curriculum_summary, metric_key, ylabel, title, output_path):
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    for short, records, _summary in curriculum_stage_records(curriculum_summary):
        x = np.linspace(0.0, 1.0, len(records))
        values = np.asarray([record[metric_key] for record in records], dtype=float)
        smooth = moving_average(values, window=5)
        color = CURRICULUM_COLORS[short]
        ax.plot(x, values, color=color, alpha=0.18, linewidth=1.2)
        ax.plot(x, smooth, color=color, linewidth=2.8, label=short)

    prettify_axes(
        ax,
        xlabel="Normalized Training Progress",
        ylabel=ylabel,
        title=title,
    )
    if "Rate" in ylabel:
        ax.set_ylim(0.0, 1.02)
    ax.legend(loc="lower right", frameon=True)
    save_figure(fig, output_path)


def plot_curriculum_sequential(curriculum_summary, output_path):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    cumulative = 0
    xticks = []
    xticklabels = []
    spans = []

    for short, records, _summary in curriculum_stage_records(curriculum_summary):
        x_local = np.arange(len(records), dtype=float)
        x = cumulative + x_local
        win = np.asarray([record["eval_win_rate"] for record in records], dtype=float)
        reward = np.asarray([record["eval_average_episode_rewards"] for record in records], dtype=float)
        color = CURRICULUM_COLORS[short]

        axes[0].plot(x, win, color=color, alpha=0.14, linewidth=1.1)
        axes[0].plot(x, moving_average(win, 5), color=color, linewidth=2.6, label=short)
        axes[1].plot(x, reward, color=color, alpha=0.14, linewidth=1.1)
        axes[1].plot(x, moving_average(reward, 5), color=color, linewidth=2.6, label=short)

        spans.append((cumulative, cumulative + len(records) - 1, color))
        xticks.append(cumulative + max(len(records) - 1, 0) / 2.0)
        xticklabels.append(short)
        cumulative += len(records)

    for idx, (start, end, color) in enumerate(spans):
        shade_alpha = 0.035 if idx % 2 == 0 else 0.055
        for ax in axes:
            ax.axvspan(start, end, color=color, alpha=shade_alpha, linewidth=0)

    prettify_axes(
        axes[0],
        ylabel="Evaluation Win Rate",
        title="Sequential Curriculum Learning Win Rate Across Stages",
    )
    axes[0].set_ylim(0.0, 1.02)
    axes[0].legend(loc="lower right", frameon=True)

    prettify_axes(
        axes[1],
        xlabel="Cumulative Evaluation Checkpoint Index",
        ylabel="Evaluation Episode Reward",
        title="Sequential Curriculum Learning Reward Across Stages",
    )
    axes[1].legend(loc="lower right", frameon=True)
    axes[1].set_xticks(xticks, xticklabels)
    save_figure(fig, output_path)


def plot_curriculum_sequential_single(curriculum_summary, value_key, ylabel, output_path, rate_ylim=False):
    fig, ax = plt.subplots(figsize=(11.6, 5.6))
    cumulative = 0
    spans = []
    stage_boundaries = []

    for short, records, summary in curriculum_stage_records(curriculum_summary):
        x_local = np.arange(len(records), dtype=float)
        x = cumulative + x_local
        values = np.asarray([record[value_key] for record in records], dtype=float)
        color = CURRICULUM_COLORS[short]

        ax.plot(x, values, color=color, alpha=0.14, linewidth=1.0)
        ax.plot(x, moving_average(values, 5), color=color, linewidth=2.7, label=short)

        stage_len = len(records)
        start = cumulative
        end = cumulative + stage_len - 1
        spans.append((start, end, color))
        stage_boundaries.append(
            {
                "short": short,
                "end": end,
                "episodes": int(summary.get("num_train_records", 0)),
            }
        )
        cumulative += stage_len

    for idx, (start, end, color) in enumerate(spans):
        shade_alpha = 0.035 if idx % 2 == 0 else 0.055
        ax.axvspan(start, end, color=color, alpha=shade_alpha, linewidth=0)

    boundary_ticks = [0]
    boundary_labels = ["Start"]
    for boundary in stage_boundaries:
        boundary_ticks.append(boundary["end"])
        boundary_labels.append(f"After {boundary['short']}\n({boundary['episodes']} episodes)")

    prettify_axes(
        ax,
        xlabel="Sequential Evaluation Checkpoint Index",
        ylabel=ylabel,
        title=None,
    )
    if rate_ylim:
        ax.set_ylim(0.0, 1.02)
    ax.set_xticks(boundary_ticks, boundary_labels)
    ax.legend(loc="lower right", frameon=True)
    save_figure(fig, output_path)


def aligned_variant_curves(ablation_summary, value_key):
    aligned = {}
    reference_x = None
    for variant_name, payload in ablation_summary["variants"].items():
        run_x = []
        run_y = []
        for run in payload["runs"]:
            records = select_eval_records(Path(run["metrics_path"]))
            if not records:
                continue
            x = np.linspace(0.0, 1.0, len(records))
            y = np.asarray([record[value_key] for record in records], dtype=float)
            run_x.append(x)
            run_y.append(y)
        if not run_x:
            continue
        min_len = min(len(item) for item in run_x)
        x = run_x[0][:min_len]
        stack = np.stack([item[:min_len] for item in run_y], axis=0)
        aligned[variant_name] = (x, stack)
        if reference_x is None or len(x) > len(reference_x):
            reference_x = x
    return aligned


def plot_ablation_single(ablation_summary, value_key, ylabel, title, output_path, ylim=None):
    fig, ax = plt.subplots(figsize=(10.4, 5.8))
    aligned = aligned_variant_curves(ablation_summary, value_key)
    for variant_name in ["full", "w_o_attention", "w_o_progressive", "baseline"]:
        if variant_name not in aligned:
            continue
        x, stack = aligned[variant_name]
        color = ABLATION_COLORS[variant_name]
        label = ABLATION_LABELS[variant_name]
        for row in stack:
            ax.plot(x, row, color=color, alpha=0.10, linewidth=0.9)
        smooth_stack = np.stack([moving_average(row, 5) for row in stack], axis=0)
        mean = smooth_stack.mean(axis=0)
        std = smooth_stack.std(axis=0)
        ax.plot(x, mean, color=color, linewidth=2.7, label=label)
        ax.fill_between(x, np.maximum(mean - std, 0.0), np.minimum(mean + std, 1.0 if ylim else mean + std), color=color, alpha=0.12)

    prettify_axes(
        ax,
        xlabel="Normalized Stage-2 Fine-Tuning Progress",
        ylabel=ylabel,
        title=title,
    )
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(loc="upper left", frameon=True)
    save_figure(fig, output_path)


def plot_benchmark_outcome(benchmark_summary, output_path):
    matchups = benchmark_summary["matchups"]
    labels = [item.get("short_label", item["label"]) for item in matchups]
    x = np.arange(len(matchups), dtype=float)
    width = 0.24

    win = np.asarray([item["aggregate"]["win_rate"] for item in matchups], dtype=float)
    draw = np.asarray([item["aggregate"]["draw_rate"] for item in matchups], dtype=float)
    loss = np.asarray([item["aggregate"]["loss_rate"] for item in matchups], dtype=float)
    win_std = np.asarray([item["aggregate"]["win_rate_std"] for item in matchups], dtype=float)
    draw_std = np.asarray([item["aggregate"]["draw_rate_std"] for item in matchups], dtype=float)
    loss_std = np.asarray([item["aggregate"]["loss_rate_std"] for item in matchups], dtype=float)

    fig, ax = plt.subplots(figsize=(9.8, 5.6))
    ax.bar(x - width, win, width=width, yerr=win_std, capsize=4, color="#4c956c", label="Win rate")
    ax.bar(x, draw, width=width, yerr=draw_std, capsize=4, color="#f1c232", label="Draw rate")
    ax.bar(x + width, loss, width=width, yerr=loss_std, capsize=4, color="#c43d3d", label="Loss rate")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, 1.02)
    prettify_axes(
        ax,
        ylabel="Outcome Rate",
        title="Benchmark Outcome Comparison",
    )
    ax.legend(loc="upper right", frameon=True)
    save_figure(fig, output_path)


def plot_benchmark_steps(benchmark_summary, output_path):
    matchups = benchmark_summary["matchups"]
    labels = [item.get("short_label", item["label"]) for item in matchups]
    x = np.arange(len(matchups), dtype=float)
    steps = np.asarray([item["aggregate"]["avg_steps"] for item in matchups], dtype=float)
    steps_std = np.asarray([item["aggregate"]["avg_steps_std"] for item in matchups], dtype=float)

    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    ax.bar(x, steps, yerr=steps_std, capsize=4, width=0.55, color="#2f6db3")
    ax.set_xticks(x, labels)
    prettify_axes(
        ax,
        ylabel="Average Episode Steps",
        title="Benchmark Episode Length",
    )
    save_figure(fig, output_path)


def build_key_metrics(curriculum_summary, ablation_summary, benchmark_summary, output_path):
    lines = []
    lines.append("# Figure Key Metrics")
    lines.append("")
    lines.append("## Curriculum Learning")
    lines.append("| Stage | Final win rate | Last-5 win rate (mean +- std) | Final reward | Last-5 reward (mean +- std) | Shield rate |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for stage in curriculum_summary["stages"]:
        summary = stage["summary"]
        lines.append(
            "| {stage} | {final_win:.3f} | {last_win:.3f} +- {last_win_std:.3f} | {final_reward:.3f} | {last_reward:.3f} +- {last_reward_std:.3f} | {shield:.3f} |".format(
                stage=stage_short_name(stage["stage_name"]),
                final_win=summary["final_win_rate"],
                last_win=summary["final_window_win_rate_mean"],
                last_win_std=summary["final_window_win_rate_std"],
                final_reward=summary["final_reward"],
                last_reward=summary["final_window_reward_mean"],
                last_reward_std=summary["final_window_reward_std"],
                shield=summary["final_safety_shield_rate"],
            )
        )
    lines.append("")
    lines.append("## Ablation Study")
    lines.append("| Variant | Final win rate (mean +- std) | Final reward (mean +- std) | Low-altitude failure rate (mean +- std) | Timeout rate (mean +- std) |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for variant_name in ["full", "w_o_attention", "w_o_progressive", "baseline"]:
        aggregate = ablation_summary["variants"][variant_name]["aggregate"]
        lines.append(
            "| {label} | {win:.3f} +- {win_std:.3f} | {reward:.3f} +- {reward_std:.3f} | {low:.3f} +- {low_std:.3f} | {timeout:.3f} +- {timeout_std:.3f} |".format(
                label=ABLATION_LABELS[variant_name],
                win=aggregate["final_win_rate_mean"],
                win_std=aggregate["final_win_rate_std"],
                reward=aggregate["final_reward_mean"],
                reward_std=aggregate["final_reward_std"],
                low=aggregate["final_low_altitude_rate_mean"],
                low_std=aggregate["final_low_altitude_rate_std"],
                timeout=aggregate["final_timeout_rate_mean"],
                timeout_std=aggregate["final_timeout_rate_std"],
            )
        )
    lines.append("")
    lines.append("## Benchmark Comparison")
    lines.append("| Matchup | Episodes | Win rate | Loss rate | Draw rate | Avg steps | Red-side win rate | Blue-side win rate |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for matchup in benchmark_summary["matchups"]:
        aggregate = matchup["aggregate"]
        lines.append(
            "| {label} | {episodes} | {win:.3f} +- {win_std:.3f} | {loss:.3f} +- {loss_std:.3f} | {draw:.3f} +- {draw_std:.3f} | {steps:.2f} +- {steps_std:.2f} | {red:.3f} | {blue:.3f} |".format(
                label=matchup.get("short_label", matchup["label"]),
                episodes=matchup["episodes"],
                win=aggregate["win_rate"],
                win_std=aggregate["win_rate_std"],
                loss=aggregate["loss_rate"],
                loss_std=aggregate["loss_rate_std"],
                draw=aggregate["draw_rate"],
                draw_std=aggregate["draw_rate_std"],
                steps=aggregate["avg_steps"],
                steps_std=aggregate["avg_steps_std"],
                red=aggregate["red_side_win_rate"],
                blue=aggregate["blue_side_win_rate"],
            )
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def plot_seed_ablation_win_rate(ablation_summary, output_path):
    variant_order = ["full", "w_o_attention", "w_o_progressive", "baseline"]
    seed_ids = sorted({int(run["seed"]) for payload in ablation_summary["variants"].values() for run in payload["runs"]})
    x = np.arange(len(seed_ids), dtype=float)
    width = 0.18

    fig, ax = plt.subplots(figsize=(10.4, 5.8))
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(variant_order))
    for offset, variant_name in zip(offsets, variant_order):
        runs = {int(run["seed"]): run for run in ablation_summary["variants"][variant_name]["runs"]}
        values = np.asarray([runs[seed]["summary"]["final_win_rate"] for seed in seed_ids], dtype=float)
        ax.bar(
            x + offset,
            values,
            width=width,
            color=ABLATION_COLORS[variant_name],
            label=ABLATION_LABELS[variant_name],
            alpha=0.88,
        )

    ax.set_xticks(x, [f"Seed {seed}" for seed in seed_ids])
    ax.set_ylim(0.0, 1.02)
    prettify_axes(ax, xlabel="Random Seed", ylabel="Final Evaluation Win Rate", title="Random-Seed Sensitivity of Ablation Variants")
    ax.legend(loc="upper left", ncol=2, frameon=True, fontsize=9)
    save_figure(fig, output_path)


def plot_seed_ablation_reward(ablation_summary, output_path):
    variant_order = ["full", "w_o_attention", "w_o_progressive", "baseline"]
    seed_ids = sorted({int(run["seed"]) for payload in ablation_summary["variants"].values() for run in payload["runs"]})
    x = np.arange(len(seed_ids), dtype=float)
    width = 0.18

    fig, ax = plt.subplots(figsize=(10.4, 5.8))
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(variant_order))
    for offset, variant_name in zip(offsets, variant_order):
        runs = {int(run["seed"]): run for run in ablation_summary["variants"][variant_name]["runs"]}
        values = np.asarray([runs[seed]["summary"]["final_reward"] for seed in seed_ids], dtype=float)
        ax.bar(
            x + offset,
            values,
            width=width,
            color=ABLATION_COLORS[variant_name],
            label=ABLATION_LABELS[variant_name],
            alpha=0.88,
        )

    ax.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--", alpha=0.75)
    ax.set_xticks(x, [f"Seed {seed}" for seed in seed_ids])
    prettify_axes(ax, xlabel="Random Seed", ylabel="Final Evaluation Reward", title="Random-Seed Sensitivity of Final Reward")
    ax.legend(loc="upper left", ncol=2, frameon=True, fontsize=9)
    save_figure(fig, output_path)


def plot_seed_ablation_distribution(ablation_summary, output_path):
    variant_order = ["full", "w_o_attention", "w_o_progressive", "baseline"]
    labels = ["Full", "No Attention", "No Progressive", "Baseline"]
    win_data = [np.asarray([run["summary"]["final_win_rate"] for run in ablation_summary["variants"][name]["runs"]], dtype=float) for name in variant_order]

    fig, ax = plt.subplots(figsize=(9.8, 5.6))
    box = ax.boxplot(
        win_data,
        patch_artist=True,
        labels=labels,
        widths=0.55,
        medianprops=dict(color="black", linewidth=1.4),
        whiskerprops=dict(color="#444444"),
        capprops=dict(color="#444444"),
    )
    for patch, variant_name in zip(box["boxes"], variant_order):
        patch.set_facecolor(ABLATION_COLORS[variant_name])
        patch.set_alpha(0.35)
    for idx, values in enumerate(win_data, start=1):
        jitter = np.linspace(-0.05, 0.05, len(values))
        ax.scatter(np.full(len(values), idx) + jitter, values, color="#333333", s=28, zorder=3)

    ax.set_ylim(0.0, 1.02)
    prettify_axes(ax, ylabel="Final Evaluation Win Rate", title="Distribution of Final Win Rate Across Random Seeds")
    save_figure(fig, output_path)


def plot_external_ippo_seed(external_summary, output_path):
    ippo_payload = next((item for item in external_summary.get("baselines", []) if item.get("variant_name") == "ippo"), None)
    if ippo_payload is None:
        return
    runs = sorted(ippo_payload["runs"], key=lambda item: int(item["seed"]))
    labels = [f"Seed {run['seed']}" for run in runs]
    x = np.arange(len(runs), dtype=float)
    win = np.asarray([run["summary"]["final_win_rate"] for run in runs], dtype=float)
    reward = np.asarray([run["summary"]["final_reward"] for run in runs], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(8.8, 7.0), sharex=True)
    axes[0].bar(x, win, width=0.55, color="#2f6db3")
    axes[0].set_ylim(0.0, 1.02)
    prettify_axes(axes[0], ylabel="Final Evaluation Win Rate", title="External IPPO Baseline Across Random Seeds")

    axes[1].bar(x, reward, width=0.55, color="#c43d3d")
    axes[1].axhline(0.0, color="#666666", linewidth=1.0, linestyle="--", alpha=0.75)
    prettify_axes(axes[1], xlabel="Random Seed", ylabel="Final Evaluation Reward", title="External IPPO Reward Across Random Seeds")
    axes[1].set_xticks(x, labels)

    save_figure(fig, output_path)


def build_seed_metrics(ablation_summary, external_summary, output_path):
    lines = []
    lines.append("# Random Seed Experiment Metrics")
    lines.append("")
    lines.append("This section only summarizes experiments that were actually repeated across random seeds.")
    lines.append("The curriculum main pipeline is not included here because the current final curriculum result is a single retained run rather than a full repeated-seed study.")
    lines.append("")
    lines.append("## Ablation Study Per-Seed Results")
    lines.append("| Variant | Seed | Final win rate | Final reward | Low-altitude failure rate | Timeout rate |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for variant_name in ["full", "w_o_attention", "w_o_progressive", "baseline"]:
        for run in sorted(ablation_summary["variants"][variant_name]["runs"], key=lambda item: int(item["seed"])):
            summary = run["summary"]
            lines.append(
                "| {label} | {seed} | {win:.3f} | {reward:.3f} | {low:.3f} | {timeout:.3f} |".format(
                    label=ABLATION_LABELS[variant_name],
                    seed=int(run["seed"]),
                    win=summary["final_win_rate"],
                    reward=summary["final_reward"],
                    low=summary["final_low_altitude_rate"],
                    timeout=summary["final_timeout_rate"],
                )
            )
    lines.append("")
    lines.append("## Ablation Aggregate Statistics")
    lines.append("| Variant | Final win rate (mean +- std) | Final reward (mean +- std) |")
    lines.append("| --- | ---: | ---: |")
    for variant_name in ["full", "w_o_attention", "w_o_progressive", "baseline"]:
        aggregate = ablation_summary["variants"][variant_name]["aggregate"]
        lines.append(
            "| {label} | {win:.3f} +- {win_std:.3f} | {reward:.3f} +- {reward_std:.3f} |".format(
                label=ABLATION_LABELS[variant_name],
                win=aggregate["final_win_rate_mean"],
                win_std=aggregate["final_win_rate_std"],
                reward=aggregate["final_reward_mean"],
                reward_std=aggregate["final_reward_std"],
            )
        )
    lines.append("")
    lines.append("## External IPPO Per-Seed Results")
    lines.append("| Baseline | Seed | Final win rate | Final reward |")
    lines.append("| --- | ---: | ---: | ---: |")
    ippo_payload = next((item for item in external_summary.get("baselines", []) if item.get("variant_name") == "ippo"), None)
    if ippo_payload is not None:
        for run in sorted(ippo_payload["runs"], key=lambda item: int(item["seed"])):
            summary = run["summary"]
            lines.append(
                "| Independent PPO | {seed} | {win:.3f} | {reward:.3f} |".format(
                    seed=int(run["seed"]),
                    win=summary["final_win_rate"],
                    reward=summary["final_reward"],
                )
            )
        aggregate = ippo_payload["aggregate"]
        lines.append("")
        lines.append("## External IPPO Aggregate Statistics")
        lines.append("| Baseline | Final win rate (mean +- std) | Final reward (mean +- std) |")
        lines.append("| --- | ---: | ---: |")
        lines.append(
            "| Independent PPO | {win:.3f} +- {win_std:.3f} | {reward:.3f} +- {reward_std:.3f} |".format(
                win=aggregate["final_win_rate_mean"],
                win_std=aggregate["final_win_rate_std"],
                reward=aggregate["final_reward_mean"],
                reward_std=aggregate["final_reward_std"],
            )
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ensure_dir(OUTPUT_ROOT)
    curriculum_summary = load_json(CURRICULUM_SUMMARY_PATH)
    ablation_summary = load_json(ABLATION_SUMMARY_PATH)
    benchmark_summary = load_json(BENCHMARK_SUMMARY_PATH)
    matd3_benchmark_summary = load_json(MATD3_BENCHMARK_SUMMARY_PATH) if MATD3_BENCHMARK_SUMMARY_PATH.exists() else {"matchups": []}
    benchmark_summary = merged_benchmark_summary(benchmark_summary, matd3_benchmark_summary)
    external_summary = load_json(EXTERNAL_BASELINE_SUMMARY_PATH) if EXTERNAL_BASELINE_SUMMARY_PATH.exists() else {"baselines": []}

    plot_curriculum_single(
        curriculum_summary,
        metric_key="eval_win_rate",
        ylabel="Evaluation Win Rate",
        title="Curriculum Learning Win Rate",
        output_path=OUTPUT_ROOT / "curriculum_win_rate.png",
    )
    plot_curriculum_single(
        curriculum_summary,
        metric_key="eval_average_episode_rewards",
        ylabel="Evaluation Episode Reward",
        title="Curriculum Learning Episode Reward",
        output_path=OUTPUT_ROOT / "curriculum_episode_reward.png",
    )
    plot_curriculum_sequential(
        curriculum_summary,
        output_path=OUTPUT_ROOT / "curriculum_sequential_dynamics.png",
    )
    plot_curriculum_sequential_single(
        curriculum_summary,
        value_key="eval_win_rate",
        ylabel="Evaluation Win Rate",
        output_path=OUTPUT_ROOT / "curriculum_sequential_win_rate.png",
        rate_ylim=True,
    )
    plot_curriculum_sequential_single(
        curriculum_summary,
        value_key="eval_average_episode_rewards",
        ylabel="Evaluation Episode Reward",
        output_path=OUTPUT_ROOT / "curriculum_sequential_reward.png",
        rate_ylim=False,
    )

    plot_ablation_single(
        ablation_summary,
        value_key="eval_win_rate",
        ylabel="Evaluation Win Rate",
        title="Ablation Study Win Rate",
        output_path=OUTPUT_ROOT / "ablation_win_rate.png",
        ylim=(0.0, 1.02),
    )
    plot_ablation_single(
        ablation_summary,
        value_key="eval_low_altitude_rate",
        ylabel="Low-Altitude Failure Rate",
        title="Ablation Study Low-Altitude Failure Rate",
        output_path=OUTPUT_ROOT / "ablation_low_altitude_failure_rate.png",
        ylim=(0.0, 1.02),
    )

    plot_benchmark_outcome(
        benchmark_summary,
        output_path=OUTPUT_ROOT / "benchmark_outcome_comparison.png",
    )
    plot_benchmark_steps(
        benchmark_summary,
        output_path=OUTPUT_ROOT / "benchmark_episode_length.png",
    )

    build_key_metrics(
        curriculum_summary,
        ablation_summary,
        benchmark_summary,
        output_path=OUTPUT_ROOT / "figure_key_metrics.md",
    )
    plot_seed_ablation_win_rate(
        ablation_summary,
        output_path=OUTPUT_ROOT / "seed_ablation_win_rate.png",
    )
    plot_seed_ablation_reward(
        ablation_summary,
        output_path=OUTPUT_ROOT / "seed_ablation_reward.png",
    )
    plot_seed_ablation_distribution(
        ablation_summary,
        output_path=OUTPUT_ROOT / "seed_ablation_win_rate_distribution.png",
    )
    plot_external_ippo_seed(
        external_summary,
        output_path=OUTPUT_ROOT / "seed_external_ippo_performance.png",
    )
    build_seed_metrics(
        ablation_summary,
        external_summary,
        output_path=OUTPUT_ROOT / "seed_key_metrics.md",
    )

    print(f"Saved paper figures to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
