import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a structured IPPO baseline and benchmark it against the final proposed curriculum policy."
    )
    parser.add_argument("--seed", type=int, default=1, help="Single random seed used for the new baseline.")
    parser.add_argument("--pretrain-scenario-name", type=str, default="2v2/ResearchHandcrafted/Stage1")
    parser.add_argument("--finetune-scenario-name", type=str, default="2v2/ResearchHandcrafted/Stage2")
    parser.add_argument("--experiment-prefix", type=str, default="external_structured_ippo_s1s2")
    parser.add_argument("--pretrain-env-steps", type=float, default=1.5e6)
    parser.add_argument("--finetune-env-steps", type=float, default=3e6)
    parser.add_argument("--versus-episodes", type=int, default=100)
    parser.add_argument("--handcrafted-episodes", type=int, default=0)
    parser.add_argument("--n-rollout-threads", type=int, default=8)
    parser.add_argument("--n-eval-rollout-threads", type=int, default=1)
    parser.add_argument("--buffer-size", type=int, default=512)
    parser.add_argument("--num-mini-batch", type=int, default=4)
    parser.add_argument("--ppo-epoch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=12)
    parser.add_argument("--progress-bar-width", type=int, default=96)
    parser.add_argument("--cuda", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    analysis_output_dir = repo_root / "scripts" / "results" / "analysis" / "benchmark_structured_ippo"

    train_command = [
        sys.executable,
        str(repo_root / "scripts" / "train_research_external_baselines.py"),
        "--baselines", "ippo_structured",
        "--seeds", str(args.seed),
        "--pretrain-scenario-name", args.pretrain_scenario_name,
        "--finetune-scenario-name", args.finetune_scenario_name,
        "--experiment-prefix", args.experiment_prefix,
        "--pretrain-env-steps", str(args.pretrain_env_steps),
        "--finetune-env-steps", str(args.finetune_env_steps),
        "--n-rollout-threads", str(args.n_rollout_threads),
        "--n-eval-rollout-threads", str(args.n_eval_rollout_threads),
        "--buffer-size", str(args.buffer_size),
        "--num-mini-batch", str(args.num_mini_batch),
        "--ppo-epoch", str(args.ppo_epoch),
        "--lr", str(args.lr),
        "--eval-interval", str(args.eval_interval),
        "--eval-episodes", str(args.eval_episodes),
        "--progress-bar-width", str(args.progress_bar_width),
        "--analyze-after",
    ]
    if args.cuda:
        train_command.append("--cuda")

    benchmark_command = [
        sys.executable,
        str(repo_root / "scripts" / "eval_research_benchmark.py"),
        "--versus-episodes", str(args.versus_episodes),
        "--handcrafted-episodes", str(args.handcrafted_episodes),
        "--external-baseline-summary",
        str(repo_root / "scripts" / "results" / "analysis" / args.experiment_prefix / "external_baselines_summary.json"),
        "--include-matchups", "ippo_structured",
        "--output-dir", str(analysis_output_dir),
    ]
    if args.cuda:
        benchmark_command.append("--cuda")

    print("[1/2] Training structured IPPO baseline")
    print("Command:")
    print(" ".join(train_command))
    subprocess.run(train_command, check=True, cwd=repo_root)

    print("")
    print("[2/2] Benchmarking against the final proposed policy")
    print("Command:")
    print(" ".join(benchmark_command))
    subprocess.run(benchmark_command, check=True, cwd=repo_root)


if __name__ == "__main__":
    main()
