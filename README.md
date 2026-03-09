# Continual Adapter Normalization

Continual learning for LLMs through adapter-based methods that reduce catastrophic forgetting by constraining update directions with respect to previous tasks.

## Methods

- **IncLoRA** (baseline): One LoRA adapter per task, freeze previous ones, no constraints. Each task gets its own isolated adapter — zero forgetting by design, but no knowledge transfer.
- **OLoRA**: Same structure but with an orthogonality loss that encourages new adapter subspaces to be orthogonal to past ones, enabling knowledge sharing while reducing interference.
- **ELLA**: Accumulates past LoRA updates into a single W_past matrix, then penalizes alignment between the current update and high-energy past directions. Lightweight, no architectural expansion, enables forward transfer through low-energy subspace reuse.

All methods use [minLoRA](https://github.com/cccntu/minLoRA)'s `torch.nn.utils.parametrize` approach for clean LoRA injection.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Run all three methods and compare results
python3 run.py --all

# Run a single method
python3 run.py --config configs/inclora.yaml
python3 run.py --config configs/olora.yaml
python3 run.py --config configs/ella.yaml

# Custom output directory
python3 run.py --config configs/ella.yaml --output_dir results/my_experiment
```

Results are saved as JSON in the output directory. When using `--all`, a comparison table is printed at the end.

## Project Structure

```
src/
  methods/
    inclora.py              # IncLoRA: multi-adapter baseline
    olora.py                # OLoRA: orthogonal LoRA with dual matrices
    ella.py                 # ELLA: energy-based alignment penalty
  training/
    continual_trainer.py    # Sequential training loop, evaluation, metrics
  data/
    datasets.py             # Load & preprocess 5 CL benchmark datasets
  utils/
    config.py               # Config dataclasses and YAML loader
  minlora/                  # minLoRA library (LoRA via parametrize)
configs/
  inclora.yaml              # IncLoRA hyperparameters
  olora.yaml                # OLoRA hyperparameters
  ella.yaml                 # ELLA hyperparameters
run.py                      # Main entry point
```

## Datasets

5 text classification benchmarks from HuggingFace, trained sequentially:

1. AG News (4 classes)
2. Yelp Review Full (5 classes)
3. Amazon Polarity (2 classes)
4. DBpedia 14 (14 classes)
5. Yahoo Answers Topics (10 classes)

## Configuration

Key hyperparameters in the YAML configs:

| Parameter | IncLoRA | OLoRA | ELLA |
|-----------|---------|-------|------|
| LoRA rank | 8 | 8 | 8 |
| LoRA alpha | 32 | 32 | 32 |
| Learning rate | 1e-3 | 1e-3 | 1e-3 |
| Epochs per task | 1 | 1 | 1 |
| Batch size | 2 | 2 | 2 |
| Grad accumulation | 4 | 4 | 4 |
| Max seq length | 256 | 256 | 256 |
| lambda_1 | 0.0 | 0.5 | 0.1 |
| Target modules | q_proj, v_proj | q_proj, v_proj | q_proj, v_proj |

## Metrics

- **ACC**: Average accuracy across all tasks after training on the last task
- **BWT**: Backward transfer — measures how much learning new tasks affects performance on old ones
- **R[i][j] matrix**: Accuracy on task j after training up to task i

## Base Model

Qwen2.5-1.5B (causal LM, loaded in bfloat16 with gradient checkpointing for memory efficiency).
