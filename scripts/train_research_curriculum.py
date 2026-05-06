import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional


DEFAULT_STAGES = [
    "2v2/ResearchHandcrafted/Stage1",
    "2v2/ResearchHandcrafted/Stage2",
    "2v2/ResearchHandcrafted/Stage3",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run handcrafted-opponent curriculum training sequentially.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed used for all stages.")
    parser.add_argument("--cuda", action="store_true", default=False, help="Enable CUDA training.")
    parser.add_argument("--stages", nargs="+", default=DEFAULT_STAGES, help="Scenario names to train sequentially.")
    parser.add_argument("--stage-steps", nargs="+", type=float, default=[8e6, 1.2e7, 1.6e7],
                        help="Number of env steps for each curriculum stage.")
    parser.add_argument("--experiment-prefix", type=str, default="curriculum_handcrafted",
                        help="Experiment prefix used to build per-stage experiment names.")
    parser.add_argument("--n-rollout-threads", type=int, default=16, help="Number of rollout environments.")
    parser.add_argument("--n-eval-rollout-threads", type=int, default=1, help="Number of eval environments.")
    parser.add_argument("--buffer-size", type=int, default=512, help="Replay buffer horizon.")
    parser.add_argument("--num-mini-batch", type=int, default=4, help="Number of PPO minibatches.")
    parser.add_argument("--ppo-epoch", type=int, default=4, help="Number of PPO epochs.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--eval-interval", type=int, default=5, help="Evaluation interval.")
    parser.add_argument("--eval-episodes", type=int, default=8, help="Number of evaluation episodes.")
    parser.add_argument("--progress-bar-width", type=int, default=84, help="Training progress bar width.")
    parser.add_argument("--analyze-after", action="store_true", default=False,
                        help="Run curriculum analysis automatically after training.")
    parser.add_argument("--analysis-output-dir", type=str, default=None,
                        help="Directory used by the post-training analysis script.")
    parser.add_argument("--analysis-smooth-window", type=int, default=5,
                        help="Moving-average window for post-training analysis.")
    return parser.parse_args()


def latest_run_dir(base_dir: Path) -> Path:
    candidates = [path for path in base_dir.iterdir() if path.is_dir() and path.name.startswith("run")]
    if len(candidates) == 0:
        raise RuntimeError(f"No run directories found in {base_dir}")

    def run_index(path: Path):
        try:
            return int(path.name[3:])
        except ValueError:
            return -1

    return sorted(candidates, key=run_index)[-1]


def build_train_command(repo_root: Path, args, scenario_name: str, experiment_name: str, num_env_steps: float, model_dir: Optional[Path]):
    command = [
        sys.executable,
        str(repo_root / "scripts" / "train" / "train_jsbsim.py"),
        "--env-name", "MultipleCombat",
        "--algorithm-name", "mappo",
        "--scenario-name", scenario_name,
        "--experiment-name", experiment_name,
        "--seed", str(args.seed),
        "--n-training-threads", "1",
        "--n-rollout-threads", str(args.n_rollout_threads),
        "--log-interval", "1",
        "--save-interval", "1",
        "--num-mini-batch", str(args.num_mini_batch),
        "--buffer-size", str(args.buffer_size),
        "--num-env-steps", str(num_env_steps),
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
        "--use-attention-policy",
        "--attention-embed-dim", "128",
        "--attention-num-heads", "4",
        "--attention-sparse-topk", "2",
        "--use-progressive-action-discretization",
        "--progressive-action-strides", "10 10 10 7|5 5 5 4|2 2 2 2|1 1 1 1",
        "--progressive-stage-boundaries", "0.0 0.35 0.7 0.9",
        "--progress-bar-width", str(args.progress_bar_width),
        "--use-eval",
        "--n-eval-rollout-threads", str(args.n_eval_rollout_threads),
        "--eval-interval", str(args.eval_interval),
        "--eval-episodes", str(args.eval_episodes),
    ]
    if args.cuda:
        command.append("--cuda")
    if model_dir is not None:
        command.extend(["--model-dir", str(model_dir)])
    return command


def main():
    args = parse_args()
    if len(args.stage_steps) != len(args.stages):
        raise ValueError("The number of --stage-steps values must match the number of --stages.")

    repo_root = Path(__file__).resolve().parents[1]
    results_root = repo_root / "scripts" / "results"
    previous_model_dir = None

    print("Curriculum stages:")
    for idx, (scenario, steps) in enumerate(zip(args.stages, args.stage_steps), start=1):
        print(f"  {idx}. {scenario} | steps={int(steps)}")

    for idx, (scenario_name, num_env_steps) in enumerate(zip(args.stages, args.stage_steps), start=1):
        stage_tag = scenario_name.split("/")[-1].lower()
        experiment_name = f"{args.experiment_prefix}_{idx:02d}_{stage_tag}"
        command = build_train_command(repo_root, args, scenario_name, experiment_name, num_env_steps, previous_model_dir)

        print("")
        print(f"[Stage {idx}/{len(args.stages)}] scenario={scenario_name}")
        if previous_model_dir is not None:
            print(f"Continue from: {previous_model_dir}")
        print("Command:")
        print(" ".join(command))

        subprocess.run(command, check=True, cwd=repo_root)

        run_base_dir = results_root / "MultipleCombat" / scenario_name / "mappo" / experiment_name
        previous_model_dir = latest_run_dir(run_base_dir)
        print(f"Stage {idx} completed. Latest model dir: {previous_model_dir}")

    print("")
    print("Curriculum training completed.")
    print(f"Final stage model dir: {previous_model_dir}")

    if args.analyze_after:
        analysis_command = [
            sys.executable,
            str(repo_root / "scripts" / "analysis" / "analyze_curriculum_results.py"),
            "--experiment-prefix", args.experiment_prefix,
            "--smooth-window", str(args.analysis_smooth_window),
        ]
        if args.analysis_output_dir:
            analysis_command.extend(["--output-dir", args.analysis_output_dir])
        print("")
        print("Running post-training curriculum analysis...")
        print("Command:")
        print(" ".join(analysis_command))
        subprocess.run(analysis_command, check=True, cwd=repo_root)


if __name__ == "__main__":
    main()
