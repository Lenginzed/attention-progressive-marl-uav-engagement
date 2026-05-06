import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Train a MATD3 external baseline and benchmark it against the final curriculum policy.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--pretrain-env-steps", type=float, default=1.5e6)
    parser.add_argument("--finetune-env-steps", type=float, default=3e6)
    parser.add_argument("--versus-episodes", type=int, default=100)
    parser.add_argument("--handcrafted-episodes", type=int, default=24)
    parser.add_argument("--cuda", action="store_true", default=False)
    parser.add_argument("--n-rollout-threads", type=int, default=8)
    parser.add_argument("--n-eval-rollout-threads", type=int, default=1)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=12)
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    train_command = [
        sys.executable,
        str(repo_root / "scripts" / "train_research_external_baselines.py"),
        "--baselines", "matd3",
        "--seeds", str(args.seed),
        "--experiment-prefix", "external_matd3_s1s2",
        "--pretrain-env-steps", str(args.pretrain_env_steps),
        "--finetune-env-steps", str(args.finetune_env_steps),
        "--n-rollout-threads", str(args.n_rollout_threads),
        "--n-eval-rollout-threads", str(args.n_eval_rollout_threads),
        "--eval-interval", str(args.eval_interval),
        "--eval-episodes", str(args.eval_episodes),
        "--analyze-after",
    ]
    if args.cuda:
        train_command.append("--cuda")

    benchmark_command = [
        sys.executable,
        str(repo_root / "scripts" / "eval_research_benchmark.py"),
        "--external-baseline-summary",
        str(repo_root / "scripts" / "results" / "analysis" / "external_matd3_s1s2" / "external_baselines_summary.json"),
        "--include-matchups", "matd3",
        "hard_handcrafted",
        "--versus-episodes", str(args.versus_episodes),
        "--handcrafted-episodes", str(args.handcrafted_episodes),
        "--output-dir",
        str(repo_root / "scripts" / "results" / "analysis" / "benchmark_matd3"),
    ]
    if args.cuda:
        benchmark_command.append("--cuda")

    print("Training MATD3 baseline...")
    print("Command:")
    print(" ".join(train_command))
    subprocess.run(train_command, check=True, cwd=repo_root)

    print("")
    print("Running MATD3 benchmark...")
    print("Command:")
    print(" ".join(benchmark_command))
    subprocess.run(benchmark_command, check=True, cwd=repo_root)


if __name__ == "__main__":
    main()
