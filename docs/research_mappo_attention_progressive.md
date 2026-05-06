# MAPPO + Multi-Dimensional Attention + Progressive Discretization for 2v2 Air Combat

## 1. Research Route

This research branch keeps the original `LAG-master` flight simulation loop and replaces the policy-side decision path with:

`state -> multi-dimensional attention encoder -> feature fusion -> MAPPO policy/value heads -> progressive discrete action selection -> simulation -> reward -> update`

The implementation maps the two reference papers into this repository as follows:

- The satellite-edge-computing paper contributes the attention front-end:
  - multi-head self-attention for global interaction modeling
  - state-to-situation cross-attention for ego-versus-opponent reasoning
  - sparse attention for salient relation filtering
  - learnable residual fusion across the three branches
- The 6-DOF air-combat paper contributes the training strategy:
  - progressive discretization of the action space
  - multi-objective reward shaping
  - curriculum-style staged training

## 2. Policy Design

### 2.1 Attention Encoder

For local observations, the encoder automatically splits the flat vector into:

- ego token: `9` dimensions
- relation tokens: `6` dimensions each for teammate / enemies / missiles when present

For centralized MAPPO critic observations, the encoder groups the joint state into per-aircraft tokens.

The final encoded feature is a weighted fusion of:

- self-attention context
- cross-attention context
- sparse-attention context

The fused representation is then passed into the original GRU + actor / critic heads.

### 2.2 Progressive Discretization

The environment action space stays fixed, but the policy only enables a stage-dependent subset of action bins.

Recommended 4-stage schedule:

1. coarse exploration: stride `10 10 10 7`
2. medium-coarse exploration: stride `5 5 5 4`
3. medium-fine control: stride `2 2 2 2`
4. fine control: stride `1 1 1 1`

Recommended stage boundaries:

1. `0.00`
2. `0.35`
3. `0.70`
4. `0.90`

This means the first `35%` of training favors fast exploration, while the last `10%` unlocks the full action precision.

## 3. Reward Design

The new research task `research_multiplecombat` uses:

- `AltitudeReward`: keeps aircraft in a safe flight envelope
- `PostureReward`: encourages angle and distance advantage
- `RelativeAltitudeReward`: discourages excessive height separation
- `TeamSpacingReward`: keeps the wingman relation in a useful support region
- `EventDrivenReward`: reserves strong sparse reward for terminal events

Recommended interpretation:

- dense tactical advantage comes from posture + relative altitude + team spacing
- survival constraint comes from altitude and bounds
- sparse mission outcome comes from event-driven reward and episode result statistics
- low-altitude safety shield can be enabled in curriculum configs to suppress unsafe dive actions during early exploration

## 4. Curriculum Design

Three staged scenarios are provided under `envs/JSBSim/configs/2v2/Research`:

- `Stage1`: small initial perturbation, shorter horizon, easiest head-on engagement
- `Stage2`: moderate perturbation, asymmetric initial geometry
- `Stage3`: large perturbation, wider engagement envelope, hardest stage

Recommended training pipeline:

1. train `Stage1` from scratch and save the checkpoint
2. initialize `Stage2` from the `Stage1` checkpoint
3. initialize `Stage3` from the `Stage2` checkpoint

Recommended opponent progression:

1. early self-play against the current latest policy under `Stage1`
2. mixed self-play using `Stage1` checkpoints while training on `Stage2`
3. evaluation and hardening against earlier-stage checkpoints under `Stage3`

### Handcrafted Opponents

An additional staged curriculum against handcrafted opponents is provided under
`envs/JSBSim/configs/2v2/ResearchHandcrafted`:

- `Stage1`: easy handcrafted opponent
- `Stage2`: medium handcrafted opponent
- `Stage3`: hard handcrafted opponent

The handcrafted opponents are implemented as rule-based high-level tactics
driving the repository's low-level flight baseline actor. Their behavior is:

- easy: conservative nearest-target pursuit with weak coordination
- medium: unique-target coordinated pursuit with spacing maintenance
- hard: focused cooperative interception with flank bias and stronger boundary recovery

### Low-Altitude Safety Shield

For the handcrafted curriculum, the stage configs can additionally enable
`use_low_altitude_safety_shield: true`.

The shield operates at environment execution time:

- clamps dangerous nose-down elevator commands when altitude enters a recovery band
- narrows aileron / rudder authority near the floor to reduce aggressive rolling
- enforces a minimum throttle during recovery
- records `ego_safety_shield_rate` in `episode_summary` and `metrics.jsonl`

## 5. Training Observability

The runner now exports:

- long textual progress bar
- reward
- win / loss / draw rates
- low-altitude / out-of-bounds / timeout rates
- safety-shield intervention rate
- average episode steps
- evaluation summaries

All records are written to `metrics.jsonl` in the run directory.

## 6. Visualization

### 6.1 Reward / Win-Rate Curves

```bash
python scripts/analysis/plot_training_curves.py --input <run_dir>/metrics.jsonl
```

### 6.2 Flight Trajectory

First render or collect a `txt.acmi` trajectory file, then run:

```bash
python scripts/analysis/plot_acmi_trajectory.py --input <path_to_acmi>
```

This script outputs:

- top-down trajectory map
- altitude-time curve

These two figures are usually enough to explain tactical advantage transfer, cooperative geometry, and boundary events in a paper or thesis.

### 6.3 Curriculum Outcome Analysis

To automatically summarize draw-to-win/loss conversion trends and compare the three stages:

```bash
python scripts/analysis/analyze_curriculum_results.py --experiment-prefix curriculum_handcrafted
```

This script outputs:

- `draw_resolution_trends.png`: per-stage draw / win / loss / resolved-rate trend
- `stage_comparison.png`: final three-stage outcome comparison and safety comparison
- `curriculum_summary.json`: stage-by-stage metric summary for later reporting

## 7. Example Training Command

```bash
cd scripts
bash train_share_selfplay_research.sh Stage1
```

For staged continuation:

```bash
MODEL_DIR=<stage1_run_dir> bash train_share_selfplay_research.sh Stage2
MODEL_DIR=<stage2_run_dir> bash train_share_selfplay_research.sh Stage3
```

For one-click curriculum training against handcrafted opponents:

```bash
cd scripts
python train_research_curriculum.py --cuda
```

To train and automatically run the post-analysis:

```bash
cd scripts
python train_research_curriculum.py --cuda --analyze-after
```

## 8. Ablation

A dedicated ablation launcher is also provided:

```bash
python scripts/train_research_ablation.py --cuda --analyze-after
```

The default ablation uses:

- warm-up scenario: `2v2/ResearchHandcrafted/Stage1`
- controlled finetune scenario: `2v2/ResearchHandcrafted/Stage2`
- variants:
  - `full`
  - `w_o_attention`
  - `w_o_progressive`
  - `baseline`
- seeds: `1 2 3`

For details, see:

- `docs/research_ablation_design.md`

## 9. Benchmark

To compare the final curriculum policy against baseline algorithms and a hard
handcrafted adversary:

```bash
python scripts/eval_research_benchmark.py --cuda
```

The default benchmark includes:

- learned-vs-learned cross-play on `2v2/ResearchVersus/Stage2`
- opponents:
  - Vanilla MAPPO
  - MAPPO + Attention
  - MAPPO + Progressive
- adversarial robustness against `hard handcrafted opponent` on
  `2v2/ResearchHandcrafted/Stage3`

Artifacts include:

- `benchmark_summary.json`
- `benchmark_comparison.png`
- representative `txt.acmi`
- annotated trajectory figures with initial positions, trajectories, and result information

For details, see:

- `docs/research_benchmark_design.md`
