# Attention-Enhanced and Progressively Discretized Multi-Agent Reinforcement Learning for Cooperative Decision-Making in Dynamic UAV Engagement

This repository contains the code, configurations, and experiment scripts for cooperative multi-agent decision-making in dynamic UAV engagement using attention-enhanced and progressively discretized reinforcement learning.

The project is built on a JSBSim-based 6-DOF air combat simulation environment and focuses on a 2v2 cooperative engagement setting. The method introduces a multi-dimensional attention mechanism for structured state representation and a progressive action discretization strategy for coarse-to-fine maneuver learning.

## Overview

The main goal of this project is to improve multi-agent autonomous decision-making in dynamic adversarial aerial scenarios with:

- structured relational perception among ego, teammate, and opponents
- progressive refinement of discrete maneuver actions
- staged curriculum learning under increasing opponent difficulty
- systematic validation through ablation and benchmark experiments

The implementation follows a centralized-training-and-decentralized-execution multi-agent reinforcement learning framework.

## Main Features

- Multi-dimensional attention for ego-teammate-opponent state encoding
- Progressive action discretization for coarse-to-fine control learning
- Curriculum learning for staged training under handcrafted opponents
- Ablation experiments for module contribution analysis
- Benchmark evaluation against multiple learned and handcrafted opponents
- Trajectory export and visualization support for engagement analysis

## Project Structure

```text
.
├─ envs/                       # JSBSim environment and task definitions
├─ scripts/                    # Training, evaluation, plotting, and analysis scripts
├─ model/                      # Pretrained or baseline model assets if included
├─ configs/                    # Scenario and experiment configuration files
├─ results/                    # Experimental outputs (optional, may be partially excluded)
├─ README.md
└─ ...
```

## Environment

Recommended environment:

- Python 3.9 or a compatible version
- Windows or Linux
- PyTorch
- Gymnasium or compatible RL dependencies
- JSBSim-related dependencies required by this project

Example Conda setup:

```bash
conda create -n mapd-marl python=3.9
conda activate mapd-marl
```

Install the required dependencies according to your local setup and the packages used in this project.

## Training

The main training workflows are implemented in the `scripts/` directory.

Typical workflows include:

- curriculum learning
- ablation training
- benchmark evaluation
- result analysis and plotting

Example:

```bash
python scripts/train_research_curriculum.py
```

Additional training and evaluation entry points are also provided in `scripts/`.

## Evaluation

Evaluation scripts support:

- benchmark comparison against multiple opponents
- staged curriculum evaluation
- trajectory export
- figure generation for result analysis

Example:

```bash
python scripts/eval_research_benchmark.py
```

## Method Summary

The proposed method combines two main design components.

### 1. Multi-dimensional Attention

The observation is reorganized into structured semantic components, including:

- ego state
- teammate-relative state
- opponent-relative state

These components are processed through multiple attention branches to improve tactical relation modeling under partial observability.

### 2. Progressive Action Discretization

Instead of exposing all fine-grained action bins from the beginning of training, the method progressively releases finer action resolutions. This reduces early exploration difficulty and supports later-stage fine control refinement.

## Reproducibility

This repository is intended to support reproducible research. Depending on repository size and publication constraints, some large assets may be excluded from the Git history, such as:

- large checkpoints
- intermediate logs
- rendered trajectory files
- full result archives

If needed, these assets can be provided separately through releases or external storage.

## Acknowledgment

This project builds upon the Light Aircraft Game environment and related codebase. Parts of the environment and simulation framework are adapted from the Light Aircraft Game / CloseAirCombat project. We acknowledge the contribution of the original authors:

```bibtex
@misc{liu2022light,
  author = {Qihan Liu and Yuhua Jiang and Xiaoteng Ma},
  title = {Light Aircraft Game: A lightweight, scalable, gym-wrapped aircraft competitive environment},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/liuqh16/CloseAirCombat}},
}
```

Project link:

https://github.com/liuqh16/CloseAirCombat

If you use this repository in academic work, please also cite the original Light Aircraft Game project where appropriate.
