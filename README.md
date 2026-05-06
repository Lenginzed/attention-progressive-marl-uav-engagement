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
