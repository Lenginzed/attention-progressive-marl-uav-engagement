# Benchmark and Adversarial Evaluation Design

## Goal

Compare the final curriculum-trained proposed policy
(`Attention + Progressive`) against representative learning baselines and a hard
handcrafted adversary.

## Recommended Setup

### Learned-vs-Learned Benchmark

- Ego policy:
  `curriculum_handcrafted_03_stage3/run1`
- Scenario:
  `2v2/ResearchVersus/Stage2`
- Episodes per matchup:
  `32`
- Side balancing:
  - first `16` episodes: ego policy controls red side
  - last `16` episodes: ego policy controls blue side
- Control mode:
  deterministic policy execution

### Adversarial Robustness Benchmark

- Ego policy:
  `curriculum_handcrafted_03_stage3/run1`
- Opponent:
  `hard handcrafted opponent`
- Scenario:
  `2v2/ResearchHandcrafted/Stage3`
- Episodes:
  `24`
- Control mode:
  deterministic policy execution

## Baselines

The benchmark script automatically selects the strongest available seed from
the ablation summary for:

- `Vanilla MAPPO`
- `MAPPO + Attention`
- `MAPPO + Progressive`

## Outputs

Running:

```bash
python scripts/eval_research_benchmark.py --cuda
```

produces:

- `benchmark_summary.json`
- `benchmark_comparison.png`
- one representative `txt.acmi` file per matchup
- one annotated trajectory figure per matchup

Representative episodes are selected by:

1. prefer ego win
2. otherwise prefer draw
3. otherwise keep a loss
