# Ablation Design for MAPPO + Multi-Dimensional Attention + Progressive Discretization

## Goal

This ablation isolates the two proposed contributions:

- multi-dimensional attention encoder
- progressive action discretization

The controlled comparison uses the same environment, reward, safety shield,
training budget, and handcrafted opponent, changing only whether the two
modules are enabled.

## Recommended Setting

- Warm-up pretraining scenario: `2v2/ResearchHandcrafted/Stage1`
- Controlled finetuning / evaluation scenario: `2v2/ResearchHandcrafted/Stage2`
- Opponents:
  - Stage1: easy handcrafted opponent
  - Stage2: medium handcrafted opponent
- Seeds: `1, 2, 3`
- Stage1 pretraining steps per run: `1.5e6`
- Stage2 finetuning steps per run: `3e6`
- Rollout threads: `8`
- Eval episodes: `12`
- PPO core hyperparameters:
  - `buffer_size=512`
  - `ppo_epoch=4`
  - `num_mini_batch=4`
  - `lr=3e-4`

Why Stage1 -> Stage2:

- direct from-scratch Stage2 training is often dominated by basic flight-stability failure
- Stage1 gives each variant a common maneuver and survival prior
- Stage2 then becomes a cleaner controlled comparison for tactical decision quality
- this setup reduces the chance that innovation gains are hidden by unstable early exploration

## Variant Matrix

1. `full`
   MAPPO + attention + progressive discretization
2. `w_o_attention`
   MAPPO + progressive discretization
3. `w_o_progressive`
   MAPPO + attention
4. `baseline`
   vanilla MAPPO

Interpretation:

- `full vs w_o_attention`: contribution of the attention encoder
- `full vs w_o_progressive`: contribution of progressive discretization
- `full vs baseline`: joint contribution of both modules
- `w_o_attention vs baseline` and `w_o_progressive vs baseline`: standalone benefit of each module

## Training Command

```bash
python scripts/train_research_ablation.py --cuda --analyze-after
```

This runs, for each variant and seed:

1. Stage1 warm-up pretraining
2. Stage2 continuation from the Stage1 checkpoint

and automatically generates:

- `ablation_learning_curves.png`
- `ablation_final_comparison.png`
- `ablation_summary.json`

## Suggested Paper Table Metrics

- final eval win rate
- final eval draw rate
- final eval resolved rate (`win + loss`)
- best eval win rate
- final low-altitude rate
- final timeout rate

## Suggested Narrative

- If `full` outperforms `w_o_attention`, then the multi-dimensional attention
  encoder improves relational situation awareness and team-level tactical modeling.
- If `full` outperforms `w_o_progressive`, then progressive discretization
  improves early exploration and later fine control.
- If `baseline` shows higher draw rate or lower resolved rate, then the two
  proposed modules together help convert prolonged neutral engagements into
  decisive tactical outcomes.
