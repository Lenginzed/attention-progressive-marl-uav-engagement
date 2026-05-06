import argparse
import subprocess
import sys
from pathlib import Path


ABLATION_VARIANTS = {
    "full": {
        "label": "MAPPO + Attention + Progressive",
        "use_attention": True,
        "use_progressive": True,
    },
    "w_o_attention": {
        "label": "MAPPO + Progressive",
        "use_attention": False,
        "use_progressive": True,
    },
    "w_o_progressive": {
        "label": "MAPPO + Attention",
        "use_attention": True,
        "use_progressive": False,
    },
    "baseline": {
        "label": "Vanilla MAPPO",
        "use_attention": False,
        "use_progressive": False,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run ablation experiments for attention and progressive discretization.")
    parser.add_argument("--pretrain-scenario-name", type=str, default="2v2/ResearchHandcrafted/Stage1",
                        help="Warm-up pretraining scenario. Default uses the easy handcrafted opponent stage.")
    parser.add_argument("--finetune-scenario-name", type=str, default="2v2/ResearchHandcrafted/Stage2",
                        help="Controlled ablation finetuning scenario. Default uses the medium handcrafted opponent stage.")
    parser.add_argument("--variants", nargs="+", default=list(ABLATION_VARIANTS.keys()),
                        choices=list(ABLATION_VARIANTS.keys()), help="Ablation variants to train.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3],
                        help="Random seeds used for each ablation variant.")
    parser.add_argument("--experiment-prefix", type=str, default="ablation_s1s2",
                        help="Prefix used to build experiment directory names.")
    parser.add_argument("--pretrain-env-steps", type=float, default=1.5e6,
                        help="Environment steps used for Stage1 warm-up pretraining.")
    parser.add_argument("--finetune-env-steps", type=float, default=3e6,
                        help="Environment steps used for Stage2 ablation finetuning.")
    parser.add_argument("--cuda", action="store_true", default=False, help="Enable CUDA training.")
    parser.add_argument("--n-rollout-threads", type=int, default=8, help="Number of rollout environments.")
    parser.add_argument("--n-eval-rollout-threads", type=int, default=1, help="Number of eval environments.")
    parser.add_argument("--buffer-size", type=int, default=512, help="Replay buffer horizon.")
    parser.add_argument("--num-mini-batch", type=int, default=4, help="Number of PPO minibatches.")
    parser.add_argument("--ppo-epoch", type=int, default=4, help="Number of PPO epochs.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--eval-interval", type=int, default=10, help="Evaluation interval.")
    parser.add_argument("--eval-episodes", type=int, default=12, help="Number of evaluation episodes.")
    parser.add_argument("--progress-bar-width", type=int, default=96, help="Training progress bar width.")
    parser.add_argument("--analyze-after", action="store_true", default=False,
                        help="Run ablation analysis automatically after all training jobs finish.")
    parser.add_argument("--analysis-output-dir", type=str, default=None,
                        help="Directory used by the post-training ablation analysis script.")
    parser.add_argument("--analysis-smooth-window", type=int, default=3,
                        help="Moving-average window used by the post-training ablation analysis.")
    parser.add_argument("--skip-existing", action="store_true", default=False,
                        help="Skip a run when its latest run directory already contains metrics.jsonl.")
    return parser.parse_args()


def build_train_commands(repo_root, args, variant_name, seed):
    variant_cfg = ABLATION_VARIANTS[variant_name]
    pretrain_experiment_name = f"{args.experiment_prefix}_pretrain_{variant_name}_seed{seed}"
    finetune_experiment_name = f"{args.experiment_prefix}_{variant_name}_seed{seed}"
    common_command = [
        sys.executable,
        str(repo_root / "scripts" / "train" / "train_jsbsim.py"),
        "--env-name", "MultipleCombat",
        "--algorithm-name", "mappo",
        "--seed", str(seed),
        "--n-training-threads", "1",
        "--n-rollout-threads", str(args.n_rollout_threads),
        "--log-interval", "1",
        "--save-interval", "1",
        "--num-mini-batch", str(args.num_mini_batch),
        "--buffer-size", str(args.buffer_size),
        "--lr", str(args.lr),
        "--gamma", "0.99",
        "--ppo-epoch", str(args.ppo_epoch),
        "--max-grad-norm", "2",
        "--entropy-coef", "1e-3",
        "--hidden-size", "128 128",
        "--act-hidden-size", "128 128",
        "--recurrent-hidden-size", "128",
        "--recurrent-hidden-layers", "1",
        "--data-chunk-length", "8",
        "--progress-bar-width", str(args.progress_bar_width),
        "--use-eval",
        "--n-eval-rollout-threads", str(args.n_eval_rollout_threads),
        "--eval-interval", str(args.eval_interval),
        "--eval-episodes", str(args.eval_episodes),
    ]
    if variant_cfg["use_attention"]:
        common_command.extend([
            "--use-attention-policy",
            "--attention-embed-dim", "128",
            "--attention-num-heads", "4",
            "--attention-sparse-topk", "2",
        ])
    if variant_cfg["use_progressive"]:
        common_command.extend([
            "--use-progressive-action-discretization",
            "--progressive-action-strides", "10 10 10 7|5 5 5 4|2 2 2 2|1 1 1 1",
            "--progressive-stage-boundaries", "0.0 0.35 0.7 0.9",
        ])
    if args.cuda:
        common_command.append("--cuda")

    pretrain_command = common_command + [
        "--scenario-name", args.pretrain_scenario_name,
        "--experiment-name", pretrain_experiment_name,
        "--num-env-steps", str(args.pretrain_env_steps),
    ]
    finetune_command = common_command + [
        "--scenario-name", args.finetune_scenario_name,
        "--experiment-name", finetune_experiment_name,
        "--num-env-steps", str(args.finetune_env_steps),
    ]
    return pretrain_experiment_name, finetune_experiment_name, pretrain_command, finetune_command


def latest_run_dir(base_dir):
    candidates = [path for path in base_dir.iterdir() if path.is_dir() and path.name.startswith("run")]
    if len(candidates) == 0:
        return None
    return sorted(candidates, key=lambda path: int(path.name[3:]) if path.name[3:].isdigit() else -1)[-1]


def should_skip_run(results_root, scenario_name, experiment_name):
    base_dir = results_root / "MultipleCombat" / scenario_name / "mappo" / experiment_name
    latest_run = latest_run_dir(base_dir) if base_dir.exists() else None
    return latest_run is not None and (latest_run / "metrics.jsonl").exists()


def latest_model_dir(results_root, scenario_name, experiment_name):
    base_dir = results_root / "MultipleCombat" / scenario_name / "mappo" / experiment_name
    if not base_dir.exists():
        return None
    latest_run = latest_run_dir(base_dir)
    if latest_run is None:
        return None
    if not (latest_run / "actor_latest.pt").exists():
        return None
    return latest_run


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    results_root = repo_root / "scripts" / "results"
    min_required_steps = args.buffer_size * args.n_rollout_threads
    if args.pretrain_env_steps < min_required_steps:
        raise ValueError(
            f"--pretrain-env-steps must be >= buffer_size * n_rollout_threads ({min_required_steps}), "
            f"got {args.pretrain_env_steps}."
        )
    if args.finetune_env_steps < min_required_steps:
        raise ValueError(
            f"--finetune-env-steps must be >= buffer_size * n_rollout_threads ({min_required_steps}), "
            f"got {args.finetune_env_steps}."
        )

    print("Ablation setup:")
    print(f"  Pretrain scenario : {args.pretrain_scenario_name}")
    print(f"  Finetune scenario : {args.finetune_scenario_name}")
    print(f"  Variants : {', '.join(args.variants)}")
    print(f"  Seeds    : {', '.join(map(str, args.seeds))}")
    print(f"  Stage1 steps : {int(args.pretrain_env_steps)} per run")
    print(f"  Stage2 steps : {int(args.finetune_env_steps)} per run")
    print(f"  Total runs: {len(args.variants) * len(args.seeds)}")

    for variant_name in args.variants:
        print("")
        print(f"[Variant] {variant_name} | {ABLATION_VARIANTS[variant_name]['label']}")
        for seed in args.seeds:
            pretrain_experiment_name, finetune_experiment_name, pretrain_command, finetune_command = build_train_commands(
                repo_root, args, variant_name, seed
            )
            if args.skip_existing and should_skip_run(results_root, args.finetune_scenario_name, finetune_experiment_name):
                print(f"  Skip existing finetune run: {finetune_experiment_name}")
                continue

            stage1_model_dir = latest_model_dir(results_root, args.pretrain_scenario_name, pretrain_experiment_name)
            if stage1_model_dir is None or not args.skip_existing:
                print(f"  Stage1 pretrain seed {seed}: {pretrain_experiment_name}")
                print("  Command:")
                print("  " + " ".join(pretrain_command))
                subprocess.run(pretrain_command, check=True, cwd=repo_root)
                stage1_model_dir = latest_model_dir(results_root, args.pretrain_scenario_name, pretrain_experiment_name)
            else:
                print(f"  Reuse existing Stage1 pretrain: {stage1_model_dir}")

            if stage1_model_dir is None:
                raise RuntimeError(f"Could not locate Stage1 checkpoint for {pretrain_experiment_name}")

            finetune_command = finetune_command + ["--model-dir", str(stage1_model_dir)]
            print(f"  Stage2 finetune seed {seed}: {finetune_experiment_name}")
            print(f"  Continue from: {stage1_model_dir}")
            print("  Command:")
            print("  " + " ".join(finetune_command))
            subprocess.run(finetune_command, check=True, cwd=repo_root)

    if args.analyze_after:
        analysis_command = [
            sys.executable,
            str(repo_root / "scripts" / "analysis" / "analyze_ablation_results.py"),
            "--scenario-name", args.finetune_scenario_name,
            "--experiment-prefix", args.experiment_prefix,
            "--smooth-window", str(args.analysis_smooth_window),
            "--variants",
            *args.variants,
            "--seeds",
            *[str(seed) for seed in args.seeds],
        ]
        if args.analysis_output_dir:
            analysis_command.extend(["--output-dir", args.analysis_output_dir])

        print("")
        print("Running post-training ablation analysis...")
        print("Command:")
        print(" ".join(analysis_command))
        subprocess.run(analysis_command, check=True, cwd=repo_root)


if __name__ == "__main__":
    main()
