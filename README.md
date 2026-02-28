# Continual Adapter Normalization

Improving continual learning for Large Language Models (LLMs) through adapter-based methods that reduce catastrophic forgetting by constraining update directions with respect to previous tasks.

## Overview

This repository explores adapter-based continual learning strategies that:

- Restrict current task updates relative to past task subspaces
- Reduce forgetting while preserving plasticity
- Analyze the geometric properties of task updates
- Study the impact of hyperparameters (rank, regularization strength, etc.)

## Key Ideas

- Orthogonal or constrained adapter updates
- Normalization of past adapters
- Subspace-aware learning dynamics
- Stability-plasticity tradeoff analysis

## Goals

- Implement mechanisms to mitigate forgetting
- Evaluate performance across sequential tasks
- Analyze update geometry and adapter interactions

## Status

Work in progress.
