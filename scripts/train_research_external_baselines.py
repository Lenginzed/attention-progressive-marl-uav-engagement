import argparse
import subprocess
import sys
from pathlib import Path


EXTERNAL_BASELINES = {
    "ippo": {
        "label": "Independent PPO",
        "algorithm_name": "ippo",
        "use_attention": False,
        "use_progressive": False,
    },
    "ippo_structured": {
        "label": "Structured IPPO",
        "algorithm_name": "ippo",
        "use_attention": True,
        "use_progressive": True,
    },
    "matd3": {
        "label": "MATD3",
        "algorithm_name": "matd3",
        "use_attention": False,
        "use_progressive": False,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train external baselines for 2v2 research air-combat experiments.")
    parser.add_argument("--baselines", nargs="+", default=["ippo"], choices=list(EXTERNAL_BASELINES.keys()))
    parser.add_argument("--pretrain-scenario-name", type=str, default="2v2/ResearchHandcrafted/Stage1")
    parser.add_argument("--finetune-scenario-name", type=str, default="2v2/ResearchHandcrafted/Stage2")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--experiment-prefix", type=str, default="external_s1s2")
    parser.add_argument("--pretrain-env-steps", type=float, default=1.5e6)
    parser.add_argument("--finetune-env-steps", type=float, default=3e6)
    parser.add_argument("--cuda", action="store_true", default=False)
    parser.add_argument("--n-rollout-threads", type=int, default=8)
    parser.add_argument("--n-eval-rollout-threads", type=int, default=1)
    parser.add_argument("--buffer-size", type=int, default=512)
    parser.add_argument("--num-mini-batch", type=int, default=4)
    parser.add_argument("--ppo-epoch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=12)
    parser.add_argument("--progress-bar-width", type=int, default=96)
    parser.add_argument("--skip-existing", action="store_true", default=False)
    parser.add_argument("--analyze-after", action="store_true", default=False)
    return parser.parse_args()


def latest_run_dir(base_dir: Path):
    candidates = [path for path in base_dir.iterdir() if path.is_dir() and path.name.startswith("run")]
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: int(path.name[3:]) if path.name[3:].isdigit() else -1)[-1]


def latest_model_dir(results_root: Path, scenario_name: str, algorithm_name: str, experiment_name: str):
    base_dir = results_root / "MultipleCombat" / scenario_name / algorithm_name / experiment_name
    if not base_dir.exists():
        return None
    run_dir = latest_run_dir(base_dir)
    if run_dir is None or not (run_dir / "actor_latest.pt").exists():
        return None
    return run_dir


def should_skip_run(results_root: Path, scenario_name: str, algorithm_name: str, experiment_name: str):
    run_dir = latest_model_dir(results_root, scenario_name, algorithm_name, experiment_name)
    return run_dir is not None and (run_dir / "metrics.jsonl").exists()


def build_command(repo_root: Path, args, baseline_name: str, scenario_name: str, experiment_name: str, num_steps: float, model_dir=None):
    baseline = EXTERNAL_BASELINES[baseline_name]
    log_interval = 1
    save_interval = 1
    eval_interval = args.eval_interval
    if baseline["algorithm_name"] == "matd3":
        log_interval = max(int(num_steps // 32), 1)
        save_interval = max(int(num_steps // 4), 1)
        eval_interval = max(int(num_steps // 8), 1)
    command = [
        sys.executable,
        str(repo_root / "scripts" / "train" / "train_jsbsim.py"),
        "--env-name", "MultipleCombat",
        "--algorithm-name", baseline["algorithm_name"],
        "--scenario-name", scenario_name,
        "--experiment-name", experiment_name,
        "--n-training-threads", "1",
        "--n-rollout-threads", str(args.n_rollout_threads),
        "--log-interval", str(log_interval),
        "--save-interval", str(save_interval),
        "--num-mini-batch", str(args.num_mini_batch),
        "--buffer-size", str(args.buffer_size),
        "--num-env-steps", str(num_steps),
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
        "--eval-interval", str(eval_interval),
        "--eval-episodes", str(args.eval_episodes),
    ]
    if baseline["algorithm_name"] == "matd3":
        command.extend([
            "--critic-lr", str(args.lr),
            "--batch-size", "256",
            "--offpolicy-buffer-capacity", "200000",
            "--learning-starts", "5000",
            "--updates-per-step", "1",
            "--tau", "0.005",
            "--policy-noise", "0.2",
            "--noise-clip", "0.5",
            "--exploration-noise", "0.1",
            "--policy-delay", "2",
        ])
    if baseline["use_attention"]:
        command.extend([
            "--use-attention-policy",
            "--attention-embed-dim", "128",
            "--attention-num-heads", "4",
            "--attention-sparse-topk", "2",
        ])
    if baseline["use_progressive"]:
        command.extend([
            "--use-progressive-action-discretization",
            "--progressive-action-strides", "10 10 10 7|5 5 5 4|2 2 2 2|1 1 1 1",
            "--progressive-stage-boundaries", "0.0 0.35 0.7 0.9",
        ])
    if args.cuda:
        command.append("--cuda")
    if model_dir is not None:
        command.extend(["--model-dir", str(model_dir)])
    return command


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    results_root = repo_root / "scripts" / "results"

    for baseline_name in args.baselines:
        baseline = EXTERNAL_BASELINES[baseline_name]
        print("")
        print(f"[External baseline] {baseline_name} | {baseline['label']}")
        for seed in args.seeds:
            pretrain_experiment = f"{args.experiment_prefix}_{baseline_name}_pretrain_seed{seed}"
            finetune_experiment = f"{args.experiment_prefix}_{baseline_name}_seed{seed}"

            if args.skip_existing and should_skip_run(results_root, args.finetune_scenario_name, baseline["algorithm_name"], finetune_experiment):
                print(f"  Skip existing fine-tune run: {finetune_experiment}")
                continue

            pretrain_command = build_command(
                repo_root, args, baseline_name, args.pretrain_scenario_name, pretrain_experiment, args.pretrain_env_steps
            ) + ["--seed", str(seed)]
            stage1_model_dir = latest_model_dir(results_root, args.pretrain_scenario_name, baseline["algorithm_name"], pretrain_experiment)
            if stage1_model_dir is None or not args.skip_existing:
                print(f"  Stage1 pretrain seed {seed}: {pretrain_experiment}")
                print("  Command:")
                print("  " + " ".join(pretrain_command))
                subprocess.run(pretrain_command, check=True, cwd=repo_root)
                stage1_model_dir = latest_model_dir(results_root, args.pretrain_scenario_name, baseline["algorithm_name"], pretrain_experiment)

            if stage1_model_dir is None:
                raise RuntimeError(f"Could not locate Stage1 checkpoint for {pretrain_experiment}")

            finetune_command = build_command(
                repo_root, args, baseline_name, args.finetune_scenario_name, finetune_experiment, args.finetune_env_steps, model_dir=stage1_model_dir
            ) + ["--seed", str(seed)]
            print(f"  Stage2 fine-tune seed {seed}: {finetune_experiment}")
            print(f"  Continue from: {stage1_model_dir}")
            print("  Command:")
            print("  " + " ".join(finetune_command))
            subprocess.run(finetune_command, check=True, cwd=repo_root)

    if args.analyze_after:
        analysis_command = [
            sys.executable,
            str(repo_root / "scripts" / "analysis" / "analyze_external_baseline_results.py"),
            "--experiment-prefix", args.experiment_prefix,
            "--scenario-name", args.finetune_scenario_name,
            "--baselines",
            *args.baselines,
            "--seeds",
            *[str(seed) for seed in args.seeds],
        ]
        print("")
        print("Running external baseline analysis...")
        print("Command:")
        print(" ".join(analysis_command))
        subprocess.run(analysis_command, check=True, cwd=repo_root)


if __name__ == "__main__":
    main()
