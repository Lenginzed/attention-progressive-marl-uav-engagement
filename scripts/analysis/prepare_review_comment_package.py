import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))

from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv
from scripts.eval_research_benchmark import build_controller


PRIMARY_BENCHMARK = Path(
    "F:/code/LAG-master/scripts/results/analysis/benchmark_focus_vanilla_ippo1_hard100_pretty/benchmark_summary.json"
)
MATD3_BENCHMARK = Path(
    "F:/code/LAG-master/scripts/results/analysis/benchmark_matd3/benchmark_summary.json"
)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def merged_benchmark_summary(primary_summary, matd3_summary=None):
    matd3_summary = matd3_summary or {"matchups": []}
    primary_map = {item["name"]: item for item in primary_summary.get("matchups", [])}
    for item in matd3_summary.get("matchups", []):
        if item["name"] not in primary_map or item.get("episodes", 0) > primary_map[item["name"]].get("episodes", 0):
            primary_map[item["name"]] = item
    ordered = []
    for key in ["vs_vanilla_mappo", "vs_ippo", "vs_matd3", "vs_hard_handcrafted"]:
        if key in primary_map:
            ordered.append(primary_map[key])
    payload = dict(primary_summary)
    payload["matchups"] = ordered
    return payload


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054):
    if n == 0:
        return 0.0, 0.0
    phat = successes / n
    denom = 1.0 + z ** 2 / n
    center = (phat + z ** 2 / (2 * n)) / denom
    half = z * math.sqrt((phat * (1.0 - phat) + z ** 2 / (4 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def exact_binomial_sf(k: int, n: int, p0: float = 0.5):
    if n <= 0:
        return 1.0
    total = 0.0
    for i in range(k, n + 1):
        total += math.comb(n, i) * (p0 ** i) * ((1.0 - p0) ** (n - i))
    return min(1.0, total)


def summarize_benchmark_statistics(benchmark_summary):
    stats = []
    for matchup in benchmark_summary["matchups"]:
        agg = matchup["aggregate"]
        n = int(agg["episodes"])
        wins = int(round(agg["win_rate"] * n))
        losses = int(round(agg["loss_rate"] * n))
        draws = int(round(agg["draw_rate"] * n))
        decisive = wins + losses
        win_ci_all = wilson_interval(wins, n)
        decisive_win_rate = wins / decisive if decisive else 0.0
        decisive_ci = wilson_interval(wins, decisive) if decisive else (0.0, 0.0)
        p_one_sided = exact_binomial_sf(wins, decisive, 0.5) if decisive else 1.0
        stats.append(
            {
                "name": matchup["name"],
                "label": matchup["opponent"]["label"],
                "episodes": n,
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "win_rate": agg["win_rate"],
                "win_rate_ci95": win_ci_all,
                "decisive_episodes": decisive,
                "decisive_win_rate": decisive_win_rate,
                "decisive_win_rate_ci95": decisive_ci,
                "one_sided_binom_p_vs_0_5": p_one_sided,
                "avg_steps": agg["avg_steps"],
                "avg_steps_std": agg["avg_steps_std"],
            }
        )
    return stats


def write_benchmark_statistics(stats, out_dir: Path):
    json_path = out_dir / "benchmark_statistical_summary.json"
    md_path = out_dir / "benchmark_statistical_summary.md"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    lines = [
        "# Benchmark Statistical Summary",
        "",
        "Win-rate confidence intervals are reported using the Wilson 95% interval.",
        "The one-sided binomial p-value is computed on decisive episodes (wins vs losses only) against the null hypothesis of 50% decisive win probability.",
        "",
        "| Opponent | Episodes | W-L-D | Win rate | 95% CI | Decisive win rate | Decisive 95% CI | One-sided p vs 0.5 | Avg. steps |",
        "| --- | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: |",
    ]
    for item in stats:
        lo, hi = item["win_rate_ci95"]
        dlo, dhi = item["decisive_win_rate_ci95"]
        lines.append(
            f"| {item['label']} | {item['episodes']} | {item['wins']}-{item['losses']}-{item['draws']} | "
            f"{item['win_rate']*100:.1f}% | [{lo*100:.1f}, {hi*100:.1f}] | "
            f"{item['decisive_win_rate']*100:.1f}% | [{dlo*100:.1f}, {dhi*100:.1f}] | "
            f"{item['one_sided_binom_p_vs_0_5']:.4f} | {item['avg_steps']:.1f} ± {item['avg_steps_std']:.1f} |"
        )

    with md_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return json_path, md_path


def normalize_episode_summary(summary, ego_on_red: bool):
    if ego_on_red:
        ego_result = summary.get("result", "draw")
        return {
            "raw_result": summary.get("result", "draw"),
            "ego_result": ego_result,
            "steps": int(summary.get("steps", 0)),
            "ego_alive": int(summary.get("ego_alive", 0)),
            "enemy_alive": int(summary.get("enemy_alive", 0)),
            "ego_score": float(summary.get("ego_score", 0.0)),
            "enemy_score": float(summary.get("enemy_score", 0.0)),
            "ego_timeout_value": float(summary.get("ego_timeout_value", 0.0)),
            "enemy_timeout_value": float(summary.get("enemy_timeout_value", 0.0)),
            "ego_end_reasons": list(summary.get("ego_end_reasons", [])),
            "enemy_end_reasons": list(summary.get("enemy_end_reasons", [])),
        }
    raw = summary.get("result", "draw")
    ego_result = "draw"
    if raw == "win":
        ego_result = "loss"
    elif raw == "loss":
        ego_result = "win"
    return {
        "raw_result": raw,
        "ego_result": ego_result,
        "steps": int(summary.get("steps", 0)),
        "ego_alive": int(summary.get("enemy_alive", 0)),
        "enemy_alive": int(summary.get("ego_alive", 0)),
        "ego_score": float(summary.get("enemy_score", 0.0)),
        "enemy_score": float(summary.get("ego_score", 0.0)),
        "ego_timeout_value": float(summary.get("enemy_timeout_value", 0.0)),
        "enemy_timeout_value": float(summary.get("ego_timeout_value", 0.0)),
        "ego_end_reasons": list(summary.get("enemy_end_reasons", [])),
        "enemy_end_reasons": list(summary.get("ego_end_reasons", [])),
    }


def aggregate_records(records):
    total = max(len(records), 1)
    wins = sum(r["episode_summary"]["ego_result"] == "win" for r in records)
    losses = sum(r["episode_summary"]["ego_result"] == "loss" for r in records)
    draws = sum(r["episode_summary"]["ego_result"] == "draw" for r in records)
    steps = np.asarray([r["episode_summary"]["steps"] for r in records], dtype=float)
    timeout_end = np.asarray(
        [
            ("timeout" in r["episode_summary"].get("ego_end_reasons", []))
            or ("timeout" in r["episode_summary"].get("enemy_end_reasons", []))
            for r in records
        ],
        dtype=float,
    )
    return {
        "episodes": len(records),
        "wins": int(wins),
        "losses": int(losses),
        "draws": int(draws),
        "win_rate": wins / total,
        "loss_rate": losses / total,
        "draw_rate": draws / total,
        "avg_steps": float(steps.mean()) if len(steps) else 0.0,
        "avg_steps_std": float(steps.std(ddof=0)) if len(steps) else 0.0,
        "timeout_end_rate": float(timeout_end.mean()) if len(timeout_end) else 0.0,
    }


def run_learned_vs_learned_episode_raw(env, ego_controller, opponent_controller, ego_on_red, episode_seed):
    env.seed(episode_seed)
    obs, _ = env.reset()
    ego_controller.reset()
    opponent_controller.reset()
    red_done = np.zeros(len(env.ego_ids), dtype=bool)
    blue_done = np.zeros(len(env.enm_ids), dtype=bool)
    info = {}
    while True:
        red_obs = obs[:len(env.ego_ids)]
        blue_obs = obs[len(env.ego_ids):len(env.ego_ids) + len(env.enm_ids)]
        if ego_on_red:
            red_actions = ego_controller.act(red_obs, red_done)
            blue_actions = opponent_controller.act(blue_obs, blue_done)
        else:
            red_actions = opponent_controller.act(red_obs, red_done)
            blue_actions = ego_controller.act(blue_obs, blue_done)
        actions = np.concatenate([red_actions, blue_actions], axis=0)
        _, _, _, dones, info = env.step(actions)
        done_flags = dones.squeeze(-1).astype(bool)
        red_done = done_flags[:len(env.ego_ids)]
        blue_done = done_flags[len(env.ego_ids):len(env.ego_ids) + len(env.enm_ids)]
        if done_flags.all():
            break
    return {
        "episode_seed": int(episode_seed),
        "ego_side": "red" if ego_on_red else "blue",
        "episode_summary": normalize_episode_summary(info["episode_summary"], ego_on_red),
    }


def run_learned_vs_handcrafted_episode_raw(env, ego_controller, episode_seed):
    env.seed(episode_seed)
    obs, _ = env.reset()
    ego_controller.reset()
    done_flags = np.zeros(env.num_agents, dtype=bool)
    info = {}
    while True:
        actions = ego_controller.act(obs, done_flags)
        obs, _, _, dones, info = env.step(actions)
        done_flags = dones.squeeze(-1).astype(bool)
        if done_flags.all():
            break
    return {
        "episode_seed": int(episode_seed),
        "ego_side": "red",
        "episode_summary": normalize_episode_summary(info["episode_summary"], True),
    }


def build_matchups(benchmark_summary):
    return benchmark_summary["ego_model"], benchmark_summary["matchups"]


def run_timeout_margin_sensitivity(benchmark_summary, out_dir: Path, episodes_per_matchup: int, base_seed: int, margins):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ego_model, matchups = build_matchups(benchmark_summary)
    results = []
    nominal_margin = 0.03

    for matchup in matchups:
        matchup_result = {
            "name": matchup["name"],
            "label": matchup["opponent"]["label"],
            "mode": matchup["mode"],
            "scenario_name": matchup["scenario_name"],
            "episodes": episodes_per_matchup,
            "nominal_margin": nominal_margin,
            "margins": [],
        }
        baseline_records = None
        for margin in margins:
            env = MultipleCombatEnv(matchup["scenario_name"])
            env.config.episode_result_margin = float(margin)
            env.task.config.episode_result_margin = float(margin)
            ego_controller = build_controller(
                ego_model, env.observation_space, env.share_observation_space, env.action_space, len(env.ego_ids), device
            )
            opponent_controller = None
            if matchup["mode"] == "learned":
                opponent_controller = build_controller(
                    matchup["opponent"],
                    env.observation_space,
                    env.share_observation_space,
                    env.action_space,
                    len(env.enm_ids),
                    device,
                )
            records = []
            for ep in range(episodes_per_matchup):
                if matchup["mode"] == "learned":
                    records.append(
                        run_learned_vs_learned_episode_raw(
                            env, ego_controller, opponent_controller, ep < episodes_per_matchup // 2, base_seed + ep
                        )
                    )
                else:
                    records.append(run_learned_vs_handcrafted_episode_raw(env, ego_controller, base_seed + 1000 + ep))
            env.close()
            aggregate = aggregate_records(records)
            outcome_strings = [r["episode_summary"]["ego_result"] for r in records]
            if abs(margin - nominal_margin) < 1e-9:
                baseline_records = outcome_strings
                change_rate = 0.0
            elif baseline_records is None:
                change_rate = 0.0
            else:
                change_rate = float(np.mean([a != b for a, b in zip(outcome_strings, baseline_records)]))
            matchup_result["margins"].append(
                {
                    "margin": float(margin),
                    "aggregate": aggregate,
                    "outcome_change_rate_vs_nominal": change_rate,
                }
            )
        results.append(matchup_result)

    json_path = out_dir / "timeout_margin_sensitivity.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    plot_path = out_dir / "timeout_margin_sensitivity.png"
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, matchup in zip(axes, results):
        x = [entry["margin"] for entry in matchup["margins"]]
        win = [entry["aggregate"]["win_rate"] for entry in matchup["margins"]]
        draw = [entry["aggregate"]["draw_rate"] for entry in matchup["margins"]]
        loss = [entry["aggregate"]["loss_rate"] for entry in matchup["margins"]]
        ax.plot(x, win, marker="o", color="#2e8b57", label="Win rate")
        ax.plot(x, draw, marker="s", color="#d4a017", label="Draw rate")
        ax.plot(x, loss, marker="^", color="#c0392b", label="Loss rate")
        ax.set_title(matchup["label"])
        ax.set_xlabel("Timeout margin")
        ax.set_ylabel("Outcome rate")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3, linestyle="--")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)

    md_path = out_dir / "timeout_margin_sensitivity.md"
    lines = [
        "# Timeout Margin Sensitivity",
        "",
        f"Nominal margin in the current implementation: `{nominal_margin}`.",
        f"Limited fixed-policy evaluation was run over `{episodes_per_matchup}` episodes per matchup using the same seed set for each tested margin.",
        "",
    ]
    for matchup in results:
        lines.append(f"## {matchup['label']}")
        lines.append("")
        lines.append("| Margin | Win rate | Loss rate | Draw rate | Avg. steps | Timeout-end rate | Outcome change vs nominal |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for entry in matchup["margins"]:
            agg = entry["aggregate"]
            lines.append(
                f"| {entry['margin']:.2f} | {agg['win_rate']*100:.1f}% | {agg['loss_rate']*100:.1f}% | "
                f"{agg['draw_rate']*100:.1f}% | {agg['avg_steps']:.1f} | {agg['timeout_end_rate']*100:.1f}% | "
                f"{entry['outcome_change_rate_vs_nominal']*100:.1f}% |"
            )
        lines.append("")
    with md_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return json_path, plot_path, md_path


def main():
    parser = argparse.ArgumentParser(description="Prepare supplementary review-comment analyses.")
    parser.add_argument(
        "--output-root",
        type=str,
        default="F:/code/LAG-master/scripts/results/result_analysis/15_ReviewComments",
    )
    parser.add_argument("--episodes-per-matchup", type=int, default=24)
    parser.add_argument("--base-seed", type=int, default=2026)
    parser.add_argument("--margins", type=float, nargs="+", default=[0.00, 0.02, 0.03, 0.05])
    parser.add_argument("--skip-timeout-sensitivity", action="store_true", default=False)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    benchmark_summary = merged_benchmark_summary(load_json(PRIMARY_BENCHMARK), load_json(MATD3_BENCHMARK))
    stats = summarize_benchmark_statistics(benchmark_summary)
    write_benchmark_statistics(stats, output_root)

    if not args.skip_timeout_sensitivity:
        run_timeout_margin_sensitivity(
            benchmark_summary,
            output_root / "figures",
            args.episodes_per_matchup,
            args.base_seed,
            args.margins,
        )


if __name__ == "__main__":
    main()
