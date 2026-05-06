import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))

from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv
from scripts.eval_research_benchmark import (
    build_controller,
    run_learned_vs_handcrafted_episode,
    run_learned_vs_learned_episode,
)


def load_matchups(summary_paths):
    matchups = {}
    for summary_path in summary_paths:
        with Path(summary_path).open("r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
        ego_model = payload["ego_model"]
        for matchup in payload["matchups"]:
            existing = matchups.get(matchup["name"])
            if existing is None or matchup["episodes"] > existing["episodes"]:
                item = dict(matchup)
                item["ego_model"] = ego_model
                matchups[matchup["name"]] = item
    return list(matchups.values())


def episode_quality(record, mode):
    summary = record["episode_summary"]
    result_rank = 2 if summary["ego_result"] == "win" else (1 if summary["ego_result"] == "draw" else 0)
    full_survival = 1 if summary.get("ego_alive", 0) >= 2 else 0
    clean_kill = 1 if summary.get("enemy_alive", 9) == 0 else 0
    non_timeout = 0 if "timeout" in summary.get("ego_end_reasons", []) else 1
    step_target = 520 if mode == "learned" else 260
    step_bonus = -abs(summary.get("steps", 0) - step_target)
    return (
        result_rank,
        clean_kill,
        full_survival,
        non_timeout,
        step_bonus,
        summary.get("steps", 0),
    )


def select_gallery_records(records, mode, limit):
    ideal = [
        record for record in records
        if record["episode_summary"]["ego_result"] == "win"
        and record["episode_summary"].get("ego_alive", 0) >= 2
        and record["episode_summary"].get("enemy_alive", 99) == 0
        and record["episode_summary"].get("steps", 0) >= (250 if mode == "learned" else 120)
    ]
    pool = ideal if len(ideal) >= limit else [record for record in records if record["episode_summary"]["ego_result"] == "win"]
    ranked = sorted(pool, key=lambda item: episode_quality(item, mode), reverse=True)
    return ranked[:limit]


def rerun_episode(matchup, record, output_dir, repo_root, device):
    env = MultipleCombatEnv(matchup["scenario_name"])
    ego_model = matchup["ego_model"]
    ego_controller = build_controller(
        ego_model,
        env.observation_space,
        env.share_observation_space,
        env.action_space,
        len(env.ego_ids),
        device,
    )

    episode_tag = f"seed{record['episode_seed']}_{record['ego_side']}"
    acmi_path = output_dir / f"{episode_tag}.txt.acmi"
    metadata_path = output_dir / f"{episode_tag}_metadata.json"
    plot_path = output_dir / f"{episode_tag}_trajectory.png"
    plot_3d_path = output_dir / f"{episode_tag}_trajectory_3d.png"

    if matchup["mode"] == "learned":
        opponent_controller = build_controller(
            matchup["opponent"],
            env.observation_space,
            env.share_observation_space,
            env.action_space,
            len(env.enm_ids),
            device,
        )
        rerun_record = run_learned_vs_learned_episode(
            env,
            ego_controller,
            opponent_controller,
            record["ego_side"] == "red",
            record["episode_seed"],
            render_path=acmi_path,
        )
    else:
        rerun_record = run_learned_vs_handcrafted_episode(
            env,
            ego_controller,
            record["episode_seed"],
            render_path=acmi_path,
        )
    env.close()

    metadata = {
        "title": matchup["label"],
        "matchup_label": matchup["label"],
        "scenario_name": matchup["scenario_name"],
        "ego_side": rerun_record["ego_side"],
        "episode_summary": rerun_record["episode_summary"],
    }
    with metadata_path.open("w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, ensure_ascii=False, indent=2)

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "analysis" / "plot_annotated_acmi_trajectory.py"),
            "--input",
            str(acmi_path),
            "--output",
            str(plot_path),
            "--output-3d",
            str(plot_3d_path),
            "--metadata",
            str(metadata_path),
        ],
        check=True,
        cwd=repo_root,
    )

    return {
        "episode_seed": rerun_record["episode_seed"],
        "ego_side": rerun_record["ego_side"],
        "episode_summary": rerun_record["episode_summary"],
        "artifacts": {
            "acmi_path": str(acmi_path),
            "metadata_path": str(metadata_path),
            "plot_path": str(plot_path),
            "plot_3d_path": str(plot_3d_path),
        },
    }


def rerun_matchup(matchup, seed_start, search_episodes, gallery_size, repo_root, device):
    env = MultipleCombatEnv(matchup["scenario_name"])
    ego_model = matchup["ego_model"]
    ego_controller = build_controller(
        ego_model,
        env.observation_space,
        env.share_observation_space,
        env.action_space,
        len(env.ego_ids),
        device,
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
    for episode_idx in range(search_episodes):
        if matchup["mode"] == "learned":
            record = run_learned_vs_learned_episode(
                env,
                ego_controller,
                opponent_controller,
                episode_idx < search_episodes // 2,
                seed_start + episode_idx,
            )
        else:
            record = run_learned_vs_handcrafted_episode(
                env,
                ego_controller,
                seed_start + 1000 + episode_idx,
            )
        records.append(record)
    env.close()

    selected = select_gallery_records(records, matchup["mode"], gallery_size)
    matchup_output_dir = Path(repo_root) / "scripts" / "results" / "result_analysis" / "10_New" / "benchmark_gallery" / matchup["name"]
    matchup_output_dir.mkdir(parents=True, exist_ok=True)
    exported = [rerun_episode(matchup, record, matchup_output_dir, repo_root, device) for record in selected]

    summary_path = matchup_output_dir / "gallery_summary.json"
    with summary_path.open("w", encoding="utf-8") as file_obj:
        json.dump(
            {
                "matchup": matchup["label"],
                "scenario_name": matchup["scenario_name"],
                "search_episodes": search_episodes,
                "selected_count": len(exported),
                "selected_records": exported,
            },
            file_obj,
            ensure_ascii=False,
            indent=2,
        )
    return str(summary_path)


def main():
    parser = argparse.ArgumentParser(description="Export multiple representative benchmark trajectories.")
    parser.add_argument(
        "--summary-paths",
        type=str,
        nargs="+",
        default=[
            "F:/code/LAG-master/scripts/results/analysis/benchmark_focus_vanilla_ippo1_hard100_pretty/benchmark_summary.json",
            "F:/code/LAG-master/scripts/results/analysis/benchmark_matd3/benchmark_summary.json",
        ],
    )
    parser.add_argument("--gallery-size", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=2026)
    parser.add_argument("--search-episodes", type=int, default=100)
    parser.add_argument("--include-matchups", type=str, nargs="+", default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    matchups = load_matchups(args.summary_paths)
    if args.include_matchups:
        requested = set(args.include_matchups)
        matchups = [matchup for matchup in matchups if matchup["name"] in requested]
    summary_paths = []
    for matchup in matchups:
        summary_paths.append(rerun_matchup(matchup, args.seed_start, max(matchup["episodes"], args.search_episodes), args.gallery_size, repo_root, device))
    print("Exported gallery summaries:")
    for path in summary_paths:
        print(path)


if __name__ == "__main__":
    main()
