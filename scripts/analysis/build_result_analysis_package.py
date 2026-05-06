import csv
import json
import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(r"F:\code\LAG-master")
RESULTS_ROOT = ROOT / "scripts" / "results"
ANALYSIS_ROOT = RESULTS_ROOT / "analysis"
OUTPUT_ROOT = RESULTS_ROOT / "result_analysis"

CURRICULUM_SUMMARY_PATH = ANALYSIS_ROOT / "curriculum_handcrafted" / "curriculum_summary.json"
ABLATION_SUMMARY_PATH = ANALYSIS_ROOT / "ablation_s1s2" / "ablation_summary.json"
BENCHMARK_SUMMARY_PATH = ANALYSIS_ROOT / "benchmark_attention_progressive" / "benchmark_summary.json"
EXTERNAL_BASELINE_SUMMARY_PATH = ANALYSIS_ROOT / "external_s1s2" / "external_baselines_summary.json"


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


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt_pm(mean_value, std_value, precision=3):
    return f"{float(mean_value):.{precision}f} ± {float(std_value):.{precision}f}"


def latex_escape(value):
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def write_latex_table(path, columns, rows, caption, label):
    align = "l" + "c" * (len(columns) - 1)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{align}}}",
        r"\hline",
        " & ".join(latex_escape(col) for col in columns) + r" \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(row.get(col, "")) for col in columns) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    path.write_text("\n".join(lines), encoding="utf-8")


def stage_short_name(stage_name):
    return stage_name.split("/")[-1]


def moving_average(values, window):
    values = np.asarray(values, dtype=float)
    if len(values) == 0 or window <= 1:
        return values
    kernel = np.ones(window, dtype=float) / window
    padded = np.pad(values, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def select_eval_records(records):
    eval_records = [record for record in records if record.get("event") == "eval"]
    if eval_records:
        return eval_records
    return [record for record in records if record.get("event") == "train"]


def plot_curriculum_dynamics(curriculum_summary, output_path):
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    colors = {"Stage1": "#5b8f29", "Stage2": "#1f77b4", "Stage3": "#d62728"}
    for stage in curriculum_summary["stages"]:
        metrics_path = Path(stage["metrics_path"])
        records = select_eval_records(load_jsonl(metrics_path))
        steps = np.asarray([record["total_num_steps"] for record in records], dtype=float)
        max_steps = steps.max() if len(steps) else 1.0
        progress = steps / max_steps if max_steps > 0 else steps
        stage_name = stage_short_name(stage["stage_name"])
        color = colors.get(stage_name, None)
        win = moving_average([record.get("eval_win_rate", record.get("win_rate", 0.0)) for record in records], 3)
        reward = moving_average([record.get("eval_average_episode_rewards", record.get("average_episode_rewards", 0.0)) for record in records], 3)
        axes[0].plot(progress, win, linewidth=2.5, label=stage_name, color=color)
        axes[1].plot(progress, reward, linewidth=2.5, label=stage_name, color=color)

    axes[0].set_ylabel("Eval Win Rate")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_title("Curriculum Learning Dynamics")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend(loc="lower right")

    axes[1].set_xlabel("Normalized Training Progress")
    axes[1].set_ylabel("Eval Episode Reward")
    axes[1].grid(True, linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def plot_curriculum_final_performance(curriculum_summary, output_path):
    stages = [stage_short_name(stage["stage_name"]) for stage in curriculum_summary["stages"]]
    win = np.asarray([stage["summary"]["final_win_rate"] for stage in curriculum_summary["stages"]], dtype=float)
    reward = np.asarray([stage["summary"]["final_reward"] for stage in curriculum_summary["stages"]], dtype=float)
    shield = np.asarray([stage["summary"]["final_safety_shield_rate"] for stage in curriculum_summary["stages"]], dtype=float)
    x = np.arange(len(stages))

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    axes[0].bar(x - 0.15, win, width=0.3, color="#2f6db3", label="Final Eval Win Rate")
    axes[0].bar(x + 0.15, shield, width=0.3, color="#d9903d", label="Safety Shield Rate")
    axes[0].set_xticks(x, stages)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("Rate")
    axes[0].set_title("Curriculum Final Performance by Stage")
    axes[0].grid(True, linestyle="--", alpha=0.35, axis="y")
    axes[0].legend(loc="upper left")

    axes[1].plot(x, reward, marker="o", markersize=8, linewidth=2.5, color="#5a8f5a")
    axes[1].set_xticks(x, stages)
    axes[1].set_ylabel("Final Eval Reward")
    axes[1].set_title("Curriculum Final Reward")
    axes[1].grid(True, linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def plot_ablation_curves(ablation_summary, output_path):
    colors = {
        "full": "#c43d3d",
        "w_o_attention": "#1f77b4",
        "w_o_progressive": "#2ca02c",
        "baseline": "#7f7f7f",
    }
    labels = {
        "full": "MAPPO + Attention + Progressive",
        "w_o_attention": "MAPPO + Progressive",
        "w_o_progressive": "MAPPO + Attention",
        "baseline": "Vanilla MAPPO",
    }

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for variant_name, payload in ablation_summary["variants"].items():
        all_steps = []
        all_win = []
        all_low = []
        for run in payload["runs"]:
            records = select_eval_records(load_jsonl(Path(run["metrics_path"])))
            steps = np.asarray([record["total_num_steps"] for record in records], dtype=float)
            if len(steps) == 0:
                continue
            norm_steps = steps / steps.max()
            all_steps.append(norm_steps)
            all_win.append(moving_average([record.get("eval_win_rate", record.get("win_rate", 0.0)) for record in records], 3))
            all_low.append(moving_average([record.get("eval_low_altitude_rate", record.get("low_altitude_rate", 0.0)) for record in records], 3))
        if not all_steps:
            continue
        min_len = min(len(arr) for arr in all_steps)
        steps = all_steps[0][:min_len]
        win_stack = np.stack([arr[:min_len] for arr in all_win], axis=0)
        low_stack = np.stack([arr[:min_len] for arr in all_low], axis=0)
        for axis, stack, title in [(axes[0], win_stack, "Eval Win Rate"), (axes[1], low_stack, "Low-Altitude Failure Rate")]:
            mean = stack.mean(axis=0)
            std = stack.std(axis=0)
            axis.plot(steps, mean, linewidth=2.4, color=colors[variant_name], label=labels[variant_name])
            axis.fill_between(steps, np.clip(mean - std, 0.0, 1.0), np.clip(mean + std, 0.0, 1.0), color=colors[variant_name], alpha=0.12)
            axis.set_title(title)
            axis.grid(True, linestyle="--", alpha=0.35)
    axes[0].set_ylabel("Rate")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].legend(loc="lower right")
    axes[1].set_xlabel("Normalized Stage-2 Fine-Tuning Progress")
    axes[1].set_ylabel("Rate")
    axes[1].set_ylim(0.0, 1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def plot_ablation_summary(ablation_summary, output_path):
    variant_order = ["full", "w_o_attention", "w_o_progressive", "baseline"]
    labels = ["Full", "No Attention", "No Progressive", "Baseline"]
    x = np.arange(len(variant_order))
    width = 0.22

    win_mean = np.asarray([ablation_summary["variants"][name]["aggregate"]["final_win_rate_mean"] for name in variant_order], dtype=float)
    win_std = np.asarray([ablation_summary["variants"][name]["aggregate"]["final_win_rate_std"] for name in variant_order], dtype=float)
    best_win_mean = np.asarray([ablation_summary["variants"][name]["aggregate"]["best_win_rate_mean"] for name in variant_order], dtype=float)
    reward_mean = np.asarray([ablation_summary["variants"][name]["aggregate"]["final_reward_mean"] for name in variant_order], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    axes[0].bar(x - width / 2, win_mean, width=width, yerr=win_std, capsize=4, color="#2f6db3", label="Final Eval Win Rate")
    axes[0].bar(x + width / 2, best_win_mean, width=width, color="#c43d3d", label="Best Eval Win Rate")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("Rate")
    axes[0].set_title("Ablation Comparison on Stage-2 Evaluation")
    axes[0].grid(True, linestyle="--", alpha=0.35, axis="y")
    axes[0].legend(loc="upper right")

    axes[1].bar(x, reward_mean, width=0.45, color="#5a8f5a")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Final Eval Reward")
    axes[1].set_title("Ablation Final Reward")
    axes[1].grid(True, linestyle="--", alpha=0.35, axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def plot_benchmark_summary(benchmark_summary, output_path):
    matchups = benchmark_summary["matchups"]
    labels = ["vs Vanilla", "vs Attention", "vs Progressive", "vs Hard Rule"]
    x = np.arange(len(matchups))
    win = np.asarray([item["aggregate"]["win_rate"] for item in matchups], dtype=float)
    draw = np.asarray([item["aggregate"]["draw_rate"] for item in matchups], dtype=float)
    loss = np.asarray([item["aggregate"]["loss_rate"] for item in matchups], dtype=float)
    steps = np.asarray([item["aggregate"]["avg_steps"] for item in matchups], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    axes[0].bar(x, win, color="#2ca02c", label="Win")
    axes[0].bar(x, draw, bottom=win, color="#f1c232", label="Draw")
    axes[0].bar(x, loss, bottom=win + draw, color="#c43d3d", label="Loss")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_ylabel("Rate")
    axes[0].set_title("Head-to-Head Benchmark Outcomes")
    axes[0].grid(True, linestyle="--", alpha=0.35, axis="y")
    axes[0].legend(loc="upper right")

    axes[1].bar(x, steps, width=0.45, color="#6a8caf")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Average Episode Steps")
    axes[1].set_title("Benchmark Episode Duration")
    axes[1].grid(True, linestyle="--", alpha=0.35, axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def copy_if_exists(src, dst):
    if Path(src).exists():
        shutil.copy2(src, dst)


def build_curriculum_package(curriculum_summary, out_dir):
    ensure_dir(out_dir)
    settings_rows = [
        {
            "Stage": "Stage1",
            "Scenario": "2v2/ResearchHandcrafted/Stage1",
            "Opponent": "Easy handcrafted opponent",
            "Training steps": "8e6",
            "Max episode steps": "700",
            "Action schedule": "Progressive discretization (4 stages)",
            "Safety shield": "Enabled",
        },
        {
            "Stage": "Stage2",
            "Scenario": "2v2/ResearchHandcrafted/Stage2",
            "Opponent": "Medium handcrafted opponent",
            "Training steps": "12e6",
            "Max episode steps": "850",
            "Action schedule": "Progressive discretization (4 stages)",
            "Safety shield": "Enabled",
        },
        {
            "Stage": "Stage3",
            "Scenario": "2v2/ResearchHandcrafted/Stage3",
            "Opponent": "Hard handcrafted opponent",
            "Training steps": "16e6",
            "Max episode steps": "1000",
            "Action schedule": "Progressive discretization (4 stages)",
            "Safety shield": "Enabled",
        },
    ]
    result_rows = []
    for stage in curriculum_summary["stages"]:
        s = stage["summary"]
        result_rows.append({
            "Stage": stage_short_name(stage["stage_name"]),
            "Final eval win rate (window mean±std)": fmt_pm(
                s.get("final_window_win_rate_mean", s["final_win_rate"]),
                s.get("final_window_win_rate_std", 0.0),
            ),
            "Final eval loss rate (window mean±std)": fmt_pm(
                s.get("final_window_loss_rate_mean", s["final_loss_rate"]),
                s.get("final_window_loss_rate_std", 0.0),
            ),
            "Final eval reward (window mean±std)": fmt_pm(
                s.get("final_window_reward_mean", s["final_reward"]),
                s.get("final_window_reward_std", 0.0),
            ),
            "Best eval win rate": f"{s['best_win_rate']:.3f}",
            "Final safety shield rate (window mean±std)": fmt_pm(
                s.get("final_window_safety_shield_rate_mean", s["final_safety_shield_rate"]),
                s.get("final_window_safety_shield_rate_std", 0.0),
            ),
            "Final timeout rate": f"{s['final_timeout_rate']:.3f}",
        })
    write_csv(out_dir / "curriculum_settings.csv", settings_rows, list(settings_rows[0].keys()))
    write_csv(out_dir / "curriculum_results.csv", result_rows, list(result_rows[0].keys()))
    write_latex_table(out_dir / "curriculum_results.tex", list(result_rows[0].keys()), result_rows, "Curriculum learning results across staged handcrafted opponents.", "tab:curriculum_results")
    plot_curriculum_dynamics(curriculum_summary, out_dir / "curriculum_learning_dynamics.png")
    plot_curriculum_final_performance(curriculum_summary, out_dir / "curriculum_final_performance.png")


def build_ablation_package(ablation_summary, out_dir):
    ensure_dir(out_dir)
    settings_rows = [{
        "Controlled stage": "Stage1 pretraining -> Stage2 evaluation",
        "Pretraining steps": "1.5e6",
        "Fine-tuning steps": "3e6",
        "Seeds": "1, 2, 3",
        "Rollout threads": "8",
        "Eval episodes": "12",
        "Scenario": "2v2/ResearchHandcrafted/Stage2",
    }]
    result_rows = []
    label_map = {
        "full": "MAPPO + Attention + Progressive",
        "w_o_attention": "MAPPO + Progressive",
        "w_o_progressive": "MAPPO + Attention",
        "baseline": "Vanilla MAPPO",
    }
    for key in ["full", "w_o_attention", "w_o_progressive", "baseline"]:
        agg = ablation_summary["variants"][key]["aggregate"]
        result_rows.append({
            "Variant": label_map[key],
            "Final win rate (mean±std)": f"{agg['final_win_rate_mean']:.3f} ± {agg['final_win_rate_std']:.3f}",
            "Best win rate (mean±std)": f"{agg['best_win_rate_mean']:.3f} ± {agg['best_win_rate_std']:.3f}",
            "Final reward (mean±std)": f"{agg['final_reward_mean']:.3f} ± {agg['final_reward_std']:.3f}",
            "Final low-altitude rate": f"{agg['final_low_altitude_rate_mean']:.3f}",
            "Final timeout rate": f"{agg['final_timeout_rate_mean']:.3f}",
        })
    write_csv(out_dir / "ablation_settings.csv", settings_rows, list(settings_rows[0].keys()))
    write_csv(out_dir / "ablation_results.csv", result_rows, list(result_rows[0].keys()))
    write_latex_table(out_dir / "ablation_results.tex", list(result_rows[0].keys()), result_rows, "Ablation results for the proposed attention module and progressive discretization.", "tab:ablation_results")
    plot_ablation_curves(ablation_summary, out_dir / "ablation_learning_dynamics.png")
    plot_ablation_summary(ablation_summary, out_dir / "ablation_summary_plot.png")


def build_benchmark_package(benchmark_summary, out_dir):
    ensure_dir(out_dir)
    settings_rows = [
        {
            "Benchmark group": "Learned-vs-learned",
            "Scenario": benchmark_summary["versus_scenario_name"],
            "Episodes per matchup": "32",
            "Side balancing": "16 red + 16 blue",
            "Execution mode": "Deterministic",
        },
        {
            "Benchmark group": "Robustness vs handcrafted",
            "Scenario": benchmark_summary["handcrafted_scenario_name"],
            "Episodes per matchup": "24",
            "Side balancing": "Red ego only",
            "Execution mode": "Deterministic",
        },
    ]
    result_rows = []
    representative_dir = ensure_dir(out_dir / "representative_cases")
    for matchup in benchmark_summary["matchups"]:
        result_rows.append({
            "Matchup": matchup["label"],
            "Episodes": matchup["aggregate"]["episodes"],
            "Win rate": f"{matchup['aggregate']['win_rate']:.3f}",
            "Loss rate": f"{matchup['aggregate']['loss_rate']:.3f}",
            "Draw rate": f"{matchup['aggregate']['draw_rate']:.3f}",
            "Average steps": f"{matchup['aggregate']['avg_steps']:.1f}",
        })
        artifacts = matchup.get("artifacts", {})
        if artifacts:
            for key, filename in [("acmi_path", f"{matchup['name']}.txt.acmi"), ("metadata_path", f"{matchup['name']}_metadata.json"), ("plot_path", f"{matchup['name']}_trajectory.png")]:
                copy_if_exists(artifacts.get(key, ""), representative_dir / filename)

    write_csv(out_dir / "benchmark_settings.csv", settings_rows, list(settings_rows[0].keys()))
    write_csv(out_dir / "benchmark_results.csv", result_rows, list(result_rows[0].keys()))
    write_latex_table(out_dir / "benchmark_results.tex", list(result_rows[0].keys()), result_rows, "Head-to-head benchmark results against learned baselines and a hard handcrafted adversary.", "tab:benchmark_results")
    plot_benchmark_summary(benchmark_summary, out_dir / "benchmark_summary_plot.png")


def build_consolidated_tables(curriculum_summary, ablation_summary, benchmark_summary, out_dir):
    ensure_dir(out_dir)
    overview_rows = []
    for stage in curriculum_summary["stages"]:
        s = stage["summary"]
        overview_rows.append({
            "Experiment group": "Curriculum",
            "Setting": stage_short_name(stage["stage_name"]),
            "Key metric 1": f"Win {s['final_win_rate']:.3f}",
            "Key metric 2": f"Reward {s['final_reward']:.1f}",
            "Key metric 3": f"Shield {s['final_safety_shield_rate']:.3f}",
        })
    for key, payload in ablation_summary["variants"].items():
        agg = payload["aggregate"]
        overview_rows.append({
            "Experiment group": "Ablation",
            "Setting": payload["label"],
            "Key metric 1": f"Final win {agg['final_win_rate_mean']:.3f}",
            "Key metric 2": f"Best win {agg['best_win_rate_mean']:.3f}",
            "Key metric 3": f"Reward {agg['final_reward_mean']:.1f}",
        })
    for matchup in benchmark_summary["matchups"]:
        agg = matchup["aggregate"]
        overview_rows.append({
            "Experiment group": "Benchmark",
            "Setting": matchup["label"],
            "Key metric 1": f"Win {agg['win_rate']:.3f}",
            "Key metric 2": f"Loss {agg['loss_rate']:.3f}",
            "Key metric 3": f"Steps {agg['avg_steps']:.1f}",
        })
    write_csv(out_dir / "overall_experiment_overview.csv", overview_rows, list(overview_rows[0].keys()))
    write_latex_table(out_dir / "overall_experiment_overview.tex", list(overview_rows[0].keys()), overview_rows, "Overall experimental overview used for paper writing.", "tab:overall_overview")


def build_report(curriculum_summary, ablation_summary, benchmark_summary, out_dir):
    ensure_dir(out_dir)
    curriculum_lines = []
    for stage in curriculum_summary["stages"]:
        s = stage["summary"]
        curriculum_lines.append(
            f"- {stage_short_name(stage['stage_name'])}: final eval win rate = {s['final_win_rate']:.3f}, "
            f"final eval reward = {s['final_reward']:.1f}, safety-shield rate = {s['final_safety_shield_rate']:.3f}."
        )

    ablation_agg = ablation_summary["variants"]
    report = f"""# Result Analysis Package

## 1. Scope

This package consolidates all completed experiments in the project:

- curriculum learning
- ablation study
- benchmark / adversarial evaluation

All figures are provided in English and formatted for direct reuse in scientific writing.

## 2. Curriculum Learning

Settings:

- Stage1: easy handcrafted opponent, 8e6 training steps
- Stage2: medium handcrafted opponent, 12e6 training steps
- Stage3: hard handcrafted opponent, 16e6 training steps
- attention module: enabled
- progressive discretization: enabled
- low-altitude safety shield: enabled

Key findings:

{chr(10).join(curriculum_lines)}

Interpretation:

- the curriculum reached perfect final evaluation win rate at Stage2 and Stage3
- no final timeout-dominated behavior remained in the final checkpoints
- the safety shield stayed active, especially in later stages, indicating that safe-flight priors remained relevant during aggressive maneuvering

## 3. Ablation Study

Settings:

- Stage1 pretraining followed by Stage2 controlled evaluation
- seeds: 1, 2, 3
- variants: full / no attention / no progressive / baseline
- evaluation metric: Stage2 final evaluation performance

Key findings:

- Full model final win rate mean: {ablation_agg['full']['aggregate']['final_win_rate_mean']:.3f}
- No-attention final win rate mean: {ablation_agg['w_o_attention']['aggregate']['final_win_rate_mean']:.3f}
- No-progressive final win rate mean: {ablation_agg['w_o_progressive']['aggregate']['final_win_rate_mean']:.3f}
- Baseline final win rate mean: {ablation_agg['baseline']['aggregate']['final_win_rate_mean']:.3f}

Interpretation:

- removing attention produced the clearest drop relative to the full model
- removing progressive discretization also reduced average final win rate relative to the full model
- the baseline remained competitive in some seeds, so the ablation result should be described as "the full model achieved the strongest average performance, but the variance across seeds remains non-negligible"

## 4. Benchmark / Adversarial Evaluation

Settings:

- learned-vs-learned benchmark on `2v2/ResearchVersus/Stage2`
- hard handcrafted adversarial test on `2v2/ResearchHandcrafted/Stage3`
- representative trajectories exported as Tacview `.acmi` and annotated figures

Key findings:

"""
    for matchup in benchmark_summary["matchups"]:
        agg = matchup["aggregate"]
        report += f"- {matchup['label']}: win rate = {agg['win_rate']:.3f}, loss rate = {agg['loss_rate']:.3f}, draw rate = {agg['draw_rate']:.3f}, average steps = {agg['avg_steps']:.1f}.\n"

    report += """

Interpretation:

- the proposed final curriculum policy showed the strongest robustness against the hard handcrafted adversary
- the head-to-head benchmark exposed matchup-specific strengths and weaknesses across learned baselines
- the benchmark should therefore be reported not only with win rates, but also with representative trajectories and end-state explanations

## 5. Recommended Paper Figures

- Curriculum learning dynamics
- Curriculum final performance by stage
- Ablation learning dynamics
- Ablation summary comparison
- Benchmark summary comparison
- Representative Tacview / trajectory plots for learned-vs-learned and learned-vs-handcrafted cases

## 6. Recommended Paper Tables

- Curriculum settings table
- Curriculum results table
- Ablation settings table
- Ablation results table
- Benchmark settings table
- Benchmark results table
"""
    (out_dir / "paper_ready_summary.md").write_text(report, encoding="utf-8")


def build_inventory(curriculum_summary, ablation_summary, benchmark_summary, out_dir):
    ensure_dir(out_dir)
    inventory = {
        "source_files": {
            "curriculum_summary": str(CURRICULUM_SUMMARY_PATH),
            "ablation_summary": str(ABLATION_SUMMARY_PATH),
            "benchmark_summary": str(BENCHMARK_SUMMARY_PATH),
        },
        "output_root": str(OUTPUT_ROOT),
        "packages": {
            "curriculum": "01_curriculum",
            "ablation": "02_ablation",
            "benchmark": "03_benchmark",
            "tables": "04_tables",
            "report": "05_report",
        },
    }
    (out_dir / "inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")


def plot_benchmark_summary(benchmark_summary, output_path):
    matchups = benchmark_summary["matchups"]
    labels = [matchup["label"].replace("Attention+Progressive Curriculum vs ", "") for matchup in matchups]
    x = np.arange(len(matchups))
    win = np.asarray([item["aggregate"]["win_rate"] for item in matchups], dtype=float)
    draw = np.asarray([item["aggregate"]["draw_rate"] for item in matchups], dtype=float)
    loss = np.asarray([item["aggregate"]["loss_rate"] for item in matchups], dtype=float)
    steps = np.asarray([item["aggregate"]["avg_steps"] for item in matchups], dtype=float)
    win_std = np.asarray([item["aggregate"].get("win_rate_std", 0.0) for item in matchups], dtype=float)
    draw_std = np.asarray([item["aggregate"].get("draw_rate_std", 0.0) for item in matchups], dtype=float)
    loss_std = np.asarray([item["aggregate"].get("loss_rate_std", 0.0) for item in matchups], dtype=float)
    steps_std = np.asarray([item["aggregate"].get("avg_steps_std", 0.0) for item in matchups], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    width = 0.22
    axes[0].bar(x - width, win, width=width, yerr=win_std, capsize=4, color="#2ca02c", label="Win")
    axes[0].bar(x, draw, width=width, yerr=draw_std, capsize=4, color="#f1c232", label="Draw")
    axes[0].bar(x + width, loss, width=width, yerr=loss_std, capsize=4, color="#c43d3d", label="Loss")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_ylabel("Rate")
    axes[0].set_title("Head-to-Head Benchmark Outcomes")
    axes[0].grid(True, linestyle="--", alpha=0.35, axis="y")
    axes[0].legend(loc="upper right")

    axes[1].bar(x, steps, width=0.45, yerr=steps_std, capsize=4, color="#6a8caf")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Average Episode Steps")
    axes[1].set_title("Benchmark Episode Duration")
    axes[1].grid(True, linestyle="--", alpha=0.35, axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def build_curriculum_package(curriculum_summary, out_dir):
    ensure_dir(out_dir)
    settings_rows = [
        {
            "Stage": "Stage1",
            "Scenario": "2v2/ResearchHandcrafted/Stage1",
            "Opponent": "Easy handcrafted opponent",
            "Training steps": "8e6",
            "Max episode steps": "700",
            "Action schedule": "Progressive discretization (4 stages)",
            "Safety shield": "Enabled",
        },
        {
            "Stage": "Stage2",
            "Scenario": "2v2/ResearchHandcrafted/Stage2",
            "Opponent": "Medium handcrafted opponent",
            "Training steps": "12e6",
            "Max episode steps": "850",
            "Action schedule": "Progressive discretization (4 stages)",
            "Safety shield": "Enabled",
        },
        {
            "Stage": "Stage3",
            "Scenario": "2v2/ResearchHandcrafted/Stage3",
            "Opponent": "Hard handcrafted opponent",
            "Training steps": "16e6",
            "Max episode steps": "1000",
            "Action schedule": "Progressive discretization (4 stages)",
            "Safety shield": "Enabled",
        },
    ]
    result_rows = []
    for stage in curriculum_summary["stages"]:
        s = stage["summary"]
        result_rows.append({
            "Stage": stage_short_name(stage["stage_name"]),
            "Final eval win rate (window mean±std)": fmt_pm(
                s.get("final_window_win_rate_mean", s["final_win_rate"]),
                s.get("final_window_win_rate_std", 0.0),
            ),
            "Final eval loss rate (window mean±std)": fmt_pm(
                s.get("final_window_loss_rate_mean", s["final_loss_rate"]),
                s.get("final_window_loss_rate_std", 0.0),
            ),
            "Final eval reward (window mean±std)": fmt_pm(
                s.get("final_window_reward_mean", s["final_reward"]),
                s.get("final_window_reward_std", 0.0),
            ),
            "Best eval win rate": f"{s['best_win_rate']:.3f}",
            "Final safety shield rate (window mean±std)": fmt_pm(
                s.get("final_window_safety_shield_rate_mean", s["final_safety_shield_rate"]),
                s.get("final_window_safety_shield_rate_std", 0.0),
            ),
            "Final timeout rate": f"{s['final_timeout_rate']:.3f}",
        })
    write_csv(out_dir / "curriculum_settings.csv", settings_rows, list(settings_rows[0].keys()))
    write_csv(out_dir / "curriculum_results.csv", result_rows, list(result_rows[0].keys()))
    write_latex_table(
        out_dir / "curriculum_results.tex",
        list(result_rows[0].keys()),
        result_rows,
        "Curriculum learning results across staged handcrafted opponents.",
        "tab:curriculum_results",
    )
    plot_curriculum_dynamics(curriculum_summary, out_dir / "curriculum_learning_dynamics.png")
    plot_curriculum_final_performance(curriculum_summary, out_dir / "curriculum_final_performance.png")


def build_ablation_package(ablation_summary, out_dir):
    ensure_dir(out_dir)
    settings_rows = [{
        "Controlled stage": "Stage1 pretraining -> Stage2 evaluation",
        "Pretraining steps": "1.5e6",
        "Fine-tuning steps": "3e6",
        "Seeds": "1, 2, 3",
        "Rollout threads": "8",
        "Eval episodes": "12",
        "Scenario": "2v2/ResearchHandcrafted/Stage2",
    }]
    result_rows = []
    label_map = {
        "full": "MAPPO + Attention + Progressive",
        "w_o_attention": "MAPPO + Progressive",
        "w_o_progressive": "MAPPO + Attention",
        "baseline": "Vanilla MAPPO",
    }
    for key in ["full", "w_o_attention", "w_o_progressive", "baseline"]:
        agg = ablation_summary["variants"][key]["aggregate"]
        result_rows.append({
            "Variant": label_map[key],
            "Final win rate (mean±std)": fmt_pm(agg["final_win_rate_mean"], agg["final_win_rate_std"]),
            "Best win rate (mean±std)": fmt_pm(agg["best_win_rate_mean"], agg["best_win_rate_std"]),
            "Final reward (mean±std)": fmt_pm(agg["final_reward_mean"], agg["final_reward_std"]),
            "Final low-altitude rate": f"{agg['final_low_altitude_rate_mean']:.3f}",
            "Final timeout rate": f"{agg['final_timeout_rate_mean']:.3f}",
        })
    write_csv(out_dir / "ablation_settings.csv", settings_rows, list(settings_rows[0].keys()))
    write_csv(out_dir / "ablation_results.csv", result_rows, list(result_rows[0].keys()))
    write_latex_table(
        out_dir / "ablation_results.tex",
        list(result_rows[0].keys()),
        result_rows,
        "Ablation results for the proposed attention module and progressive discretization.",
        "tab:ablation_results",
    )
    plot_ablation_curves(ablation_summary, out_dir / "ablation_learning_dynamics.png")
    plot_ablation_summary(ablation_summary, out_dir / "ablation_summary_plot.png")


def build_benchmark_package(benchmark_summary, out_dir):
    ensure_dir(out_dir)
    learned_episodes = "50"
    handcrafted_episodes = "24"
    if benchmark_summary.get("matchups"):
        learned = [m for m in benchmark_summary["matchups"] if m.get("mode") == "learned"]
        handcrafted = [m for m in benchmark_summary["matchups"] if m.get("mode") == "handcrafted"]
        if learned:
            learned_episodes = str(learned[0]["episodes"])
        if handcrafted:
            handcrafted_episodes = str(handcrafted[0]["episodes"])
    settings_rows = [
        {
            "Benchmark group": "Learned-vs-learned",
            "Scenario": benchmark_summary["versus_scenario_name"],
            "Episodes per matchup": learned_episodes,
            "Side balancing": "Balanced red/blue assignment",
            "Execution mode": "Deterministic",
        },
        {
            "Benchmark group": "Robustness vs handcrafted",
            "Scenario": benchmark_summary["handcrafted_scenario_name"],
            "Episodes per matchup": handcrafted_episodes,
            "Side balancing": "Red ego only",
            "Execution mode": "Deterministic",
        },
    ]
    result_rows = []
    representative_dir = ensure_dir(out_dir / "representative_cases")
    for matchup in benchmark_summary["matchups"]:
        agg = matchup["aggregate"]
        result_rows.append({
            "Matchup": matchup["label"],
            "Episodes": agg["episodes"],
            "Win rate (mean±std)": fmt_pm(agg["win_rate"], agg.get("win_rate_std", 0.0)),
            "Loss rate (mean±std)": fmt_pm(agg["loss_rate"], agg.get("loss_rate_std", 0.0)),
            "Draw rate (mean±std)": fmt_pm(agg["draw_rate"], agg.get("draw_rate_std", 0.0)),
            "Average steps (mean±std)": fmt_pm(agg["avg_steps"], agg.get("avg_steps_std", 0.0), precision=1),
        })
        artifacts = matchup.get("artifacts", {})
        if artifacts:
            for key, filename in [
                ("acmi_path", f"{matchup['name']}.txt.acmi"),
                ("metadata_path", f"{matchup['name']}_metadata.json"),
                ("plot_path", f"{matchup['name']}_trajectory.png"),
            ]:
                copy_if_exists(artifacts.get(key, ""), representative_dir / filename)

    write_csv(out_dir / "benchmark_settings.csv", settings_rows, list(settings_rows[0].keys()))
    write_csv(out_dir / "benchmark_results.csv", result_rows, list(result_rows[0].keys()))
    write_latex_table(
        out_dir / "benchmark_results.tex",
        list(result_rows[0].keys()),
        result_rows,
        "Head-to-head benchmark results against learned baselines and a hard handcrafted adversary.",
        "tab:benchmark_results",
    )
    plot_benchmark_summary(benchmark_summary, out_dir / "benchmark_summary_plot.png")


def build_consolidated_tables(curriculum_summary, ablation_summary, benchmark_summary, out_dir):
    ensure_dir(out_dir)
    overview_rows = []
    for stage in curriculum_summary["stages"]:
        s = stage["summary"]
        overview_rows.append({
            "Experiment group": "Curriculum",
            "Setting": stage_short_name(stage["stage_name"]),
            "Key metric 1": f"Win {fmt_pm(s.get('final_window_win_rate_mean', s['final_win_rate']), s.get('final_window_win_rate_std', 0.0))}",
            "Key metric 2": f"Reward {fmt_pm(s.get('final_window_reward_mean', s['final_reward']), s.get('final_window_reward_std', 0.0), precision=1)}",
            "Key metric 3": f"Shield {fmt_pm(s.get('final_window_safety_shield_rate_mean', s['final_safety_shield_rate']), s.get('final_window_safety_shield_rate_std', 0.0))}",
        })
    for _, payload in ablation_summary["variants"].items():
        agg = payload["aggregate"]
        overview_rows.append({
            "Experiment group": "Ablation",
            "Setting": payload["label"],
            "Key metric 1": f"Final win {fmt_pm(agg['final_win_rate_mean'], agg['final_win_rate_std'])}",
            "Key metric 2": f"Best win {fmt_pm(agg['best_win_rate_mean'], agg['best_win_rate_std'])}",
            "Key metric 3": f"Reward {fmt_pm(agg['final_reward_mean'], agg['final_reward_std'], precision=1)}",
        })
    for matchup in benchmark_summary["matchups"]:
        agg = matchup["aggregate"]
        overview_rows.append({
            "Experiment group": "Benchmark",
            "Setting": matchup["label"],
            "Key metric 1": f"Win {fmt_pm(agg['win_rate'], agg.get('win_rate_std', 0.0))}",
            "Key metric 2": f"Loss {fmt_pm(agg['loss_rate'], agg.get('loss_rate_std', 0.0))}",
            "Key metric 3": f"Steps {fmt_pm(agg['avg_steps'], agg.get('avg_steps_std', 0.0), precision=1)}",
        })
    write_csv(out_dir / "overall_experiment_overview.csv", overview_rows, list(overview_rows[0].keys()))
    write_latex_table(out_dir / "overall_experiment_overview.tex", list(overview_rows[0].keys()), overview_rows, "Overall experimental overview used for paper writing.", "tab:overall_overview")


def build_report(curriculum_summary, ablation_summary, benchmark_summary, out_dir):
    ensure_dir(out_dir)
    curriculum_lines = []
    for stage in curriculum_summary["stages"]:
        s = stage["summary"]
        curriculum_lines.append(
            f"- {stage_short_name(stage['stage_name'])}: final-window eval win rate = "
            f"{fmt_pm(s.get('final_window_win_rate_mean', s['final_win_rate']), s.get('final_window_win_rate_std', 0.0))}, "
            f"reward = {fmt_pm(s.get('final_window_reward_mean', s['final_reward']), s.get('final_window_reward_std', 0.0), precision=1)}, "
            f"safety-shield rate = {fmt_pm(s.get('final_window_safety_shield_rate_mean', s['final_safety_shield_rate']), s.get('final_window_safety_shield_rate_std', 0.0))}."
        )

    ablation_agg = ablation_summary["variants"]
    report = f"""# Result Analysis Package

## 1. Scope

This package consolidates all completed experiments in the project:

- curriculum learning
- ablation study
- benchmark / adversarial evaluation

All figures are provided in English and formatted for direct reuse in scientific writing.

## 2. Curriculum Learning

Settings:

- Stage1: easy handcrafted opponent, 8e6 training steps
- Stage2: medium handcrafted opponent, 12e6 training steps
- Stage3: hard handcrafted opponent, 16e6 training steps
- attention module: enabled
- progressive discretization: enabled
- low-altitude safety shield: enabled

Key findings:

{chr(10).join(curriculum_lines)}

## 3. Ablation Study

Key findings:

- Full model final win rate: {fmt_pm(ablation_agg['full']['aggregate']['final_win_rate_mean'], ablation_agg['full']['aggregate']['final_win_rate_std'])}
- No-attention final win rate: {fmt_pm(ablation_agg['w_o_attention']['aggregate']['final_win_rate_mean'], ablation_agg['w_o_attention']['aggregate']['final_win_rate_std'])}
- No-progressive final win rate: {fmt_pm(ablation_agg['w_o_progressive']['aggregate']['final_win_rate_mean'], ablation_agg['w_o_progressive']['aggregate']['final_win_rate_std'])}
- Baseline final win rate: {fmt_pm(ablation_agg['baseline']['aggregate']['final_win_rate_mean'], ablation_agg['baseline']['aggregate']['final_win_rate_std'])}

## 4. Benchmark / Adversarial Evaluation

"""
    for matchup in benchmark_summary["matchups"]:
        agg = matchup["aggregate"]
        report += (
            f"- {matchup['label']}: win = {fmt_pm(agg['win_rate'], agg.get('win_rate_std', 0.0))}, "
            f"loss = {fmt_pm(agg['loss_rate'], agg.get('loss_rate_std', 0.0))}, "
            f"draw = {fmt_pm(agg['draw_rate'], agg.get('draw_rate_std', 0.0))}, "
            f"steps = {fmt_pm(agg['avg_steps'], agg.get('avg_steps_std', 0.0), precision=1)}.\n"
        )
    report += """

## 5. Recommended Paper Figures

- Curriculum learning dynamics
- Curriculum final performance by stage
- Ablation learning dynamics
- Ablation summary comparison
- Benchmark summary comparison
- Representative Tacview / trajectory plots for learned-vs-learned and learned-vs-handcrafted cases
"""
    (out_dir / "paper_ready_summary.md").write_text(report, encoding="utf-8")


def build_inventory(curriculum_summary, ablation_summary, benchmark_summary, out_dir):
    ensure_dir(out_dir)
    source_files = {
        "curriculum_summary": str(CURRICULUM_SUMMARY_PATH),
        "ablation_summary": str(ABLATION_SUMMARY_PATH),
        "benchmark_summary": str(BENCHMARK_SUMMARY_PATH),
    }
    if EXTERNAL_BASELINE_SUMMARY_PATH.exists():
        source_files["external_baseline_summary"] = str(EXTERNAL_BASELINE_SUMMARY_PATH)
    inventory = {
        "source_files": source_files,
        "output_root": str(OUTPUT_ROOT),
        "packages": {
            "curriculum": "01_curriculum",
            "ablation": "02_ablation",
            "benchmark": "03_benchmark",
            "tables": "04_tables",
            "report": "05_report",
        },
    }
    (out_dir / "inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    curriculum_summary = load_json(CURRICULUM_SUMMARY_PATH)
    ablation_summary = load_json(ABLATION_SUMMARY_PATH)
    benchmark_summary = load_json(BENCHMARK_SUMMARY_PATH)

    ensure_dir(OUTPUT_ROOT)
    build_inventory(curriculum_summary, ablation_summary, benchmark_summary, ensure_dir(OUTPUT_ROOT / "00_overview"))
    build_curriculum_package(curriculum_summary, ensure_dir(OUTPUT_ROOT / "01_curriculum"))
    build_ablation_package(ablation_summary, ensure_dir(OUTPUT_ROOT / "02_ablation"))
    build_benchmark_package(benchmark_summary, ensure_dir(OUTPUT_ROOT / "03_benchmark"))
    build_consolidated_tables(curriculum_summary, ablation_summary, benchmark_summary, ensure_dir(OUTPUT_ROOT / "04_tables"))
    build_report(curriculum_summary, ablation_summary, benchmark_summary, ensure_dir(OUTPUT_ROOT / "05_report"))
    print(f"Saved consolidated analysis package to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
