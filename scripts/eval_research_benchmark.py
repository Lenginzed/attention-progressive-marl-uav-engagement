import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import get_config
from algorithms.matd3.policy import MATD3Policy
from algorithms.mappo.ppo_policy import PPOPolicy
from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv


VARIANT_FLAGS = {
    "full": {"use_attention": True, "use_progressive": True},
    "w_o_attention": {"use_attention": False, "use_progressive": True},
    "w_o_progressive": {"use_attention": True, "use_progressive": False},
    "baseline": {"use_attention": False, "use_progressive": False},
    "curriculum_full": {"use_attention": True, "use_progressive": True},
    "ippo": {"use_attention": False, "use_progressive": False},
    "ippo_structured": {"use_attention": True, "use_progressive": True},
    "matd3": {"use_attention": False, "use_progressive": False},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run benchmark and adversarial experiments for trained research policies.")
    parser.add_argument("--ego-model-dir", type=str,
                        default="F:/code/LAG-master/scripts/results/MultipleCombat/2v2/ResearchHandcrafted/Stage3/mappo/curriculum_handcrafted_03_stage3/run1",
                        help="Directory containing the final proposed policy checkpoint.")
    parser.add_argument("--ego-label", type=str, default="Attention+Progressive Curriculum", help="Display name for the ego policy.")
    parser.add_argument("--ablation-summary", type=str,
                        default="F:/code/LAG-master/scripts/results/analysis/ablation_s1s2/ablation_summary.json",
                        help="Path to ablation summary json used to select the best baseline checkpoints.")
    parser.add_argument("--external-baseline-summary", type=str, default=None,
                        help="Optional summary json for external baselines such as IPPO.")
    parser.add_argument("--versus-scenario-name", type=str, default="2v2/ResearchVersus/Stage2",
                        help="Model-vs-model benchmark scenario.")
    parser.add_argument("--handcrafted-scenario-name", type=str, default="2v2/ResearchHandcrafted/Stage3",
                        help="Model-vs-hard-handcrafted benchmark scenario.")
    parser.add_argument("--versus-episodes", type=int, default=50, help="Episodes per learned-policy matchup.")
    parser.add_argument("--handcrafted-episodes", type=int, default=24, help="Episodes against the hard handcrafted opponent.")
    parser.add_argument("--seed", type=int, default=2026, help="Base random seed for evaluation episodes.")
    parser.add_argument("--cuda", action="store_true", default=False, help="Enable CUDA inference.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory used to store benchmark artifacts.")
    parser.add_argument("--skip-render", action="store_true", default=False, help="Skip representative Tacview rendering.")
    parser.add_argument(
        "--include-matchups",
        type=str,
        nargs="+",
        default=None,
        choices=["vanilla_mappo", "attention_only", "progressive_only", "ippo", "ippo_structured", "matd3", "hard_handcrafted"],
        help="Subset of benchmark opponents to evaluate. If omitted, all available matchups are included.",
    )
    parser.add_argument(
        "--external-baseline-seed-overrides",
        type=str,
        nargs="+",
        default=None,
        help="Optional overrides formatted as variant_name:seed, for example ippo:1.",
    )
    return parser.parse_args()


def make_policy_args(use_attention, use_progressive):
    args = get_config().parse_args([])
    args.algorithm_name = "mappo"
    args.use_attention_policy = use_attention
    args.use_progressive_action_discretization = use_progressive
    args.use_prior = False
    args.hidden_size = "128 128"
    args.act_hidden_size = "128 128"
    args.recurrent_hidden_size = 128
    args.recurrent_hidden_layers = 1
    args.use_recurrent_policy = True
    args.use_feature_normalization = False
    args.activation_id = 1
    args.gain = 0.01
    args.attention_embed_dim = 128
    args.attention_num_heads = 4
    args.attention_dropout = 0.0
    args.attention_sparse_topk = 2
    args.attention_critic_query_tokens = 2
    args.progressive_action_strides = "10 10 10 7|5 5 5 4|2 2 2 2|1 1 1 1"
    args.progressive_stage_boundaries = "0.0 0.35 0.7 0.9"
    args.lr = 3e-4
    return args


class LearnedController:
    def __init__(self, model_dir, label, variant_name, obs_space, share_obs_space, act_space, team_size, device):
        flags = VARIANT_FLAGS[variant_name]
        args = make_policy_args(flags["use_attention"], flags["use_progressive"])
        self.label = label
        self.team_size = team_size
        self.policy = PPOPolicy(args, obs_space, share_obs_space, act_space, device=device)
        actor_path = Path(model_dir) / "actor_latest.pt"
        actor_state = torch.load(str(actor_path), map_location=device, weights_only=True)
        self.policy.actor.load_state_dict(actor_state)
        self.policy.prep_rollout()
        self.policy.set_progress(1.0)
        self.reset()

    def reset(self):
        self.rnn_states = np.zeros((self.team_size, 1, 128), dtype=np.float32)
        self.masks = np.ones((self.team_size, 1), dtype=np.float32)

    def act(self, obs, done_flags):
        self.masks = np.ones_like(self.masks, dtype=np.float32)
        self.masks[np.asarray(done_flags, dtype=bool)] = 0.0
        with torch.no_grad():
            actions, self.rnn_states = self.policy.act(obs, self.rnn_states, self.masks, deterministic=True)
        return actions.detach().cpu().numpy()


class MATD3Controller:
    def __init__(self, model_dir, label, obs_space, share_obs_space, act_space, team_size, device):
        args = get_config().parse_args([])
        args.algorithm_name = "matd3"
        args.use_attention_policy = False
        args.hidden_size = "128 128"
        args.act_hidden_size = "128 128"
        args.use_feature_normalization = False
        args.activation_id = 1
        args.lr = 3e-4
        args.critic_lr = 3e-4
        self.label = label
        self.team_size = team_size
        self.policy = MATD3Policy(args, obs_space, share_obs_space, act_space, num_agents=team_size, device=device)
        actor_path = Path(model_dir) / "actor_latest.pt"
        actor_state = torch.load(str(actor_path), map_location=device, weights_only=True)
        self.policy.actor.load_state_dict(actor_state)
        self.policy.actor_target.load_state_dict(actor_state)
        self.policy.prep_rollout()

    def reset(self):
        return None

    def act(self, obs, done_flags):
        del done_flags
        return self.policy.act(obs, deterministic=True)


def build_controller(model_spec, obs_space, share_obs_space, act_space, team_size, device):
    if model_spec.get("algorithm_name") == "matd3" or model_spec.get("variant_name") == "matd3":
        return MATD3Controller(model_spec["model_dir"], model_spec["label"], obs_space, share_obs_space, act_space, team_size, device)
    return LearnedController(model_spec["model_dir"], model_spec["label"], model_spec["variant_name"], obs_space, share_obs_space, act_space, team_size, device)


def select_best_experiment(summary_path, variant_name):
    with open(summary_path, "r", encoding="utf-8") as file_obj:
        summary = json.load(file_obj)
    runs = summary["variants"][variant_name]["runs"]
    best_run = max(
        runs,
        key=lambda run: (
            float(run["summary"].get("final_win_rate", 0.0)),
            float(run["summary"].get("best_win_rate", 0.0)),
            float(run["summary"].get("final_reward", 0.0)),
        ),
    )
    return {
        "label": summary["variants"][variant_name]["label"],
        "variant_name": variant_name,
        "algorithm_name": "mappo",
        "model_dir": str(Path(best_run["metrics_path"]).parent),
        "seed": best_run["seed"],
        "experiment_name": best_run["experiment_name"],
    }


def normalize_result(result, ego_on_red):
    if ego_on_red:
        return result
    if result == "win":
        return "loss"
    if result == "loss":
        return "win"
    return "draw"


def translate_episode_summary(summary, ego_on_red):
    return {
        "raw_result": summary.get("result", "draw"),
        "ego_result": normalize_result(summary.get("result", "draw"), ego_on_red),
        "steps": int(summary.get("steps", 0)),
        "ego_alive": int(summary.get("ego_alive", 0) if ego_on_red else summary.get("enemy_alive", 0)),
        "enemy_alive": int(summary.get("enemy_alive", 0) if ego_on_red else summary.get("ego_alive", 0)),
        "ego_end_reasons": list(summary.get("ego_end_reasons", []) if ego_on_red else summary.get("enemy_end_reasons", [])),
        "enemy_end_reasons": list(summary.get("enemy_end_reasons", []) if ego_on_red else summary.get("ego_end_reasons", [])),
    }


def aggregate_episode_records(records):
    total = max(len(records), 1)
    win_series = np.asarray([record["episode_summary"]["ego_result"] == "win" for record in records], dtype=float)
    loss_series = np.asarray([record["episode_summary"]["ego_result"] == "loss" for record in records], dtype=float)
    draw_series = np.asarray([record["episode_summary"]["ego_result"] == "draw" for record in records], dtype=float)
    steps = np.asarray([record["episode_summary"]["steps"] for record in records], dtype=float)
    red_records = [record for record in records if record["ego_side"] == "red"]
    blue_records = [record for record in records if record["ego_side"] == "blue"]

    def _rate(records_subset, result_name):
        if not records_subset:
            return 0.0
        return float(np.mean([record["episode_summary"]["ego_result"] == result_name for record in records_subset]))

    return {
        "episodes": len(records),
        "win_rate": float(win_series.mean()) if len(win_series) else 0.0,
        "win_rate_std": float(win_series.std(ddof=0)) if len(win_series) else 0.0,
        "loss_rate": float(loss_series.mean()) if len(loss_series) else 0.0,
        "loss_rate_std": float(loss_series.std(ddof=0)) if len(loss_series) else 0.0,
        "draw_rate": float(draw_series.mean()) if len(draw_series) else 0.0,
        "draw_rate_std": float(draw_series.std(ddof=0)) if len(draw_series) else 0.0,
        "avg_steps": float(steps.mean()) if len(steps) else 0.0,
        "avg_steps_std": float(steps.std(ddof=0)) if len(steps) else 0.0,
        "red_side_win_rate": _rate(red_records, "win"),
        "blue_side_win_rate": _rate(blue_records, "win"),
    }


def parse_seed_overrides(override_items):
    overrides = {}
    if not override_items:
        return overrides
    for item in override_items:
        if ":" not in item:
            raise ValueError(f"Invalid seed override '{item}'. Expected variant_name:seed.")
        variant_name, seed_text = item.split(":", 1)
        overrides[variant_name.strip()] = int(seed_text)
    return overrides


def load_external_baselines(summary_path, seed_overrides=None):
    if summary_path is None:
        return []
    summary_file = Path(summary_path)
    if not summary_file.exists():
        return []
    with summary_file.open("r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)

    baselines = []
    for baseline in payload.get("baselines", []):
        selected = baseline
        override_seed = (seed_overrides or {}).get(baseline["variant_name"])
        if override_seed is not None:
            matching_runs = [run for run in baseline.get("runs", []) if int(run.get("seed", -1)) == override_seed]
            if not matching_runs:
                raise ValueError(
                    f"Requested seed override {baseline['variant_name']}:{override_seed} was not found in {summary_file}."
                )
            run = matching_runs[0]
            selected = {
                "label": baseline["label"],
                "variant_name": baseline["variant_name"],
                "model_dir": run["model_dir"],
                "seed": run["seed"],
                "experiment_name": run["experiment_name"],
            }
        baselines.append({
            "label": selected["label"],
            "variant_name": selected["variant_name"],
            "algorithm_name": baseline.get("algorithm_name", "mappo"),
            "model_dir": selected["model_dir"],
            "seed": selected.get("seed"),
            "experiment_name": selected.get("experiment_name"),
        })
    return baselines


def run_learned_vs_learned_episode(env, ego_controller, opponent_controller, ego_on_red, episode_seed, render_path=None):
    env.seed(episode_seed)
    obs, share_obs = env.reset()
    ego_controller.reset()
    opponent_controller.reset()
    if render_path is not None:
        env.render(mode="txt", filepath=str(render_path))

    red_done = np.zeros(len(env.ego_ids), dtype=bool)
    blue_done = np.zeros(len(env.enm_ids), dtype=bool)
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
        obs, share_obs, rewards, dones, info = env.step(actions)
        if render_path is not None:
            env.render(mode="txt", filepath=str(render_path))
        done_flags = dones.squeeze(-1).astype(bool)
        red_done = done_flags[:len(env.ego_ids)]
        blue_done = done_flags[len(env.ego_ids):len(env.ego_ids) + len(env.enm_ids)]
        if done_flags.all():
            break
    return {
        "episode_seed": episode_seed,
        "ego_side": "red" if ego_on_red else "blue",
        "episode_summary": translate_episode_summary(info["episode_summary"], ego_on_red),
    }


def run_learned_vs_handcrafted_episode(env, ego_controller, episode_seed, render_path=None):
    env.seed(episode_seed)
    obs, share_obs = env.reset()
    ego_controller.reset()
    if render_path is not None:
        env.render(mode="txt", filepath=str(render_path))

    done_flags = np.zeros(env.num_agents, dtype=bool)
    while True:
        actions = ego_controller.act(obs, done_flags)
        obs, share_obs, rewards, dones, info = env.step(actions)
        if render_path is not None:
            env.render(mode="txt", filepath=str(render_path))
        done_flags = dones.squeeze(-1).astype(bool)
        if done_flags.all():
            break
    return {
        "episode_seed": episode_seed,
        "ego_side": "red",
        "episode_summary": translate_episode_summary(info["episode_summary"], True),
    }


def pick_representative_record(records):
    rank = {"win": 2, "draw": 1, "loss": 0}
    return max(
        records,
        key=lambda record: (
            rank[record["episode_summary"]["ego_result"]],
            record["episode_summary"]["steps"],
            record["episode_summary"].get("ego_alive", 0),
            -record["episode_summary"].get("enemy_alive", 0),
        ),
    )


def render_representative(matchup, ego_model, output_dir, repo_root, device):
    acmi_path = output_dir / f"{matchup['name']}.txt.acmi"
    metadata_path = output_dir / f"{matchup['name']}_metadata.json"
    plot_path = output_dir / f"{matchup['name']}_trajectory.png"
    plot_3d_path = output_dir / f"{matchup['name']}_trajectory_3d.png"

    if matchup["mode"] == "learned":
        env = MultipleCombatEnv(matchup["scenario_name"])
        ego_controller = build_controller(ego_model, env.observation_space, env.share_observation_space, env.action_space, len(env.ego_ids), device)
        opponent_controller = build_controller(matchup["opponent"], env.observation_space, env.share_observation_space, env.action_space, len(env.enm_ids), device)
        record = run_learned_vs_learned_episode(env, ego_controller, opponent_controller, matchup["representative"]["ego_side"] == "red", matchup["representative"]["episode_seed"], render_path=acmi_path)
    else:
        env = MultipleCombatEnv(matchup["scenario_name"])
        ego_controller = build_controller(ego_model, env.observation_space, env.share_observation_space, env.action_space, len(env.ego_ids), device)
        record = run_learned_vs_handcrafted_episode(env, ego_controller, matchup["representative"]["episode_seed"], render_path=acmi_path)
    env.close()

    metadata = {
        "title": matchup["label"],
        "matchup_label": matchup["label"],
        "scenario_name": matchup["scenario_name"],
        "ego_side": record["ego_side"],
        "episode_summary": record["episode_summary"],
    }
    with metadata_path.open("w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, ensure_ascii=False, indent=2)

    subprocess.run([
        sys.executable,
        str(repo_root / "scripts" / "analysis" / "plot_annotated_acmi_trajectory.py"),
        "--input", str(acmi_path),
        "--output", str(plot_path),
        "--output-3d", str(plot_3d_path),
        "--metadata", str(metadata_path),
    ], check=True, cwd=repo_root)
    return {
        "acmi_path": str(acmi_path),
        "metadata_path": str(metadata_path),
        "plot_path": str(plot_path),
        "plot_3d_path": str(plot_3d_path),
    }


def plot_benchmark_summary(matchups, output_path):
    labels = [matchup["label"] for matchup in matchups]
    x = np.arange(len(labels))
    width = 0.22
    win = np.asarray([matchup["aggregate"]["win_rate"] for matchup in matchups], dtype=float)
    draw = np.asarray([matchup["aggregate"]["draw_rate"] for matchup in matchups], dtype=float)
    loss = np.asarray([matchup["aggregate"]["loss_rate"] for matchup in matchups], dtype=float)
    avg_steps = np.asarray([matchup["aggregate"]["avg_steps"] for matchup in matchups], dtype=float)
    win_std = np.asarray([matchup["aggregate"].get("win_rate_std", 0.0) for matchup in matchups], dtype=float)
    draw_std = np.asarray([matchup["aggregate"].get("draw_rate_std", 0.0) for matchup in matchups], dtype=float)
    loss_std = np.asarray([matchup["aggregate"].get("loss_rate_std", 0.0) for matchup in matchups], dtype=float)
    avg_steps_std = np.asarray([matchup["aggregate"].get("avg_steps_std", 0.0) for matchup in matchups], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    axes[0].bar(x - width, win, width=width, yerr=win_std, capsize=4, label="Win Rate")
    axes[0].bar(x, draw, width=width, yerr=draw_std, capsize=4, label="Draw Rate")
    axes[0].bar(x + width, loss, width=width, yerr=loss_std, capsize=4, label="Loss Rate")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_ylabel("Rate")
    axes[0].set_title("Benchmark Outcome Comparison")
    axes[0].grid(True, linestyle="--", alpha=0.4, axis="y")
    axes[0].legend()

    axes[1].bar(x, avg_steps, width=0.45, yerr=avg_steps_std, capsize=4, color="#5a8f5a")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Average Episode Steps")
    axes[1].set_title("Benchmark Episode Length")
    axes[1].grid(True, linestyle="--", alpha=0.4, axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir) if args.output_dir else repo_root / "scripts" / "results" / "analysis" / "benchmark_attention_progressive"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")

    ego_model = {"label": args.ego_label, "variant_name": "curriculum_full", "algorithm_name": "mappo", "model_dir": args.ego_model_dir}
    baseline = select_best_experiment(args.ablation_summary, "baseline")
    attention_only = select_best_experiment(args.ablation_summary, "w_o_progressive")
    progressive_only = select_best_experiment(args.ablation_summary, "w_o_attention")
    seed_overrides = parse_seed_overrides(args.external_baseline_seed_overrides)
    external_baselines = load_external_baselines(args.external_baseline_summary, seed_overrides)

    learned_matchup_map = {
        "vanilla_mappo": {
            "name": "vs_vanilla_mappo",
            "label": f"{args.ego_label} vs Vanilla MAPPO",
            "mode": "learned",
            "scenario_name": args.versus_scenario_name,
            "episodes": args.versus_episodes,
            "opponent": baseline,
        },
        "attention_only": {
            "name": "vs_attention_only",
            "label": f"{args.ego_label} vs MAPPO+Attention",
            "mode": "learned",
            "scenario_name": args.versus_scenario_name,
            "episodes": args.versus_episodes,
            "opponent": attention_only,
        },
        "progressive_only": {
            "name": "vs_progressive_only",
            "label": f"{args.ego_label} vs MAPPO+Progressive",
            "mode": "learned",
            "scenario_name": args.versus_scenario_name,
            "episodes": args.versus_episodes,
            "opponent": progressive_only,
        },
    }
    for external in external_baselines:
        learned_matchup_map[external["variant_name"]] = {
            "name": f"vs_{external['variant_name']}",
            "label": f"{args.ego_label} vs {external['label']}",
            "mode": "learned",
            "scenario_name": args.versus_scenario_name,
            "episodes": args.versus_episodes,
            "opponent": external,
        }
    handcrafted_matchup = {
        "name": "vs_hard_handcrafted",
        "label": f"{args.ego_label} vs Hard Handcrafted",
        "mode": "handcrafted",
        "scenario_name": args.handcrafted_scenario_name,
        "episodes": args.handcrafted_episodes,
        "opponent": {"label": "Hard Handcrafted Opponent", "variant_name": "handcrafted"},
    }

    requested = set(args.include_matchups) if args.include_matchups else None
    learned_matchups = []
    for matchup_key in ["vanilla_mappo", "attention_only", "progressive_only"]:
        if requested is None or matchup_key in requested:
            learned_matchups.append(learned_matchup_map[matchup_key])
    for external in external_baselines:
        matchup_key = external["variant_name"]
        if requested is None or matchup_key in requested:
            learned_matchups.append(learned_matchup_map[matchup_key])

    benchmark_results = []
    for matchup in learned_matchups:
        env = MultipleCombatEnv(matchup["scenario_name"])
        ego_controller = build_controller(ego_model, env.observation_space, env.share_observation_space, env.action_space, len(env.ego_ids), device)
        opponent_controller = build_controller(matchup["opponent"], env.observation_space, env.share_observation_space, env.action_space, len(env.enm_ids), device)
        records = []
        for episode_idx in range(matchup["episodes"]):
            records.append(run_learned_vs_learned_episode(env, ego_controller, opponent_controller, episode_idx < matchup["episodes"] // 2, args.seed + episode_idx))
        env.close()
        matchup["records"] = records
        matchup["aggregate"] = aggregate_episode_records(records)
        matchup["representative"] = pick_representative_record(records)
        benchmark_results.append(matchup)

    if requested is None or "hard_handcrafted" in requested:
        env = MultipleCombatEnv(handcrafted_matchup["scenario_name"])
        ego_controller = build_controller(ego_model, env.observation_space, env.share_observation_space, env.action_space, len(env.ego_ids), device)
        handcrafted_records = []
        for episode_idx in range(handcrafted_matchup["episodes"]):
            handcrafted_records.append(run_learned_vs_handcrafted_episode(env, ego_controller, args.seed + 1000 + episode_idx))
        env.close()
        handcrafted_matchup["records"] = handcrafted_records
        handcrafted_matchup["aggregate"] = aggregate_episode_records(handcrafted_records)
        handcrafted_matchup["representative"] = pick_representative_record(handcrafted_records)
        benchmark_results.append(handcrafted_matchup)

    if not args.skip_render:
        for matchup in benchmark_results:
            matchup["artifacts"] = render_representative(matchup, ego_model, output_dir, repo_root, device)

    figure_path = output_dir / "benchmark_comparison.png"
    summary_path = output_dir / "benchmark_summary.json"
    plot_benchmark_summary(benchmark_results, figure_path)

    with summary_path.open("w", encoding="utf-8") as file_obj:
        json.dump({
            "ego_model": ego_model,
            "versus_scenario_name": args.versus_scenario_name,
            "handcrafted_scenario_name": args.handcrafted_scenario_name,
            "benchmark_figure": str(figure_path),
            "matchups": [{
                "name": matchup["name"],
                "label": matchup["label"],
                "mode": matchup["mode"],
                "scenario_name": matchup["scenario_name"],
                "episodes": matchup["episodes"],
                "opponent": matchup["opponent"],
                "aggregate": matchup["aggregate"],
                "representative": matchup["representative"],
                "artifacts": matchup.get("artifacts", {}),
            } for matchup in benchmark_results],
        }, file_obj, ensure_ascii=False, indent=2)

    print(f"Saved benchmark figure to {figure_path}")
    print(f"Saved benchmark summary to {summary_path}")


if __name__ == "__main__":
    main()
