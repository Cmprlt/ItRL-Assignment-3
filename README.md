# ItRL-Assignment-3

DQN agents for CartPole that are trained to balance poles of different lengths.
Every strategy changes only which pole length is used in each training episode.
The network, DQN algorithm, and hyperparameters stay the same so results can be compared directly. Standard strategy runners use seeds 0-9 with 1000 episodes per seed.

## Setup

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Project layout

```
├── config/       # Training configuration
├── models/       # Q-network architecture
├── samplers/     # Pole-length strategies and diagnostics
├── training/     # Shared DQN training implementation
├── runners/      # Strategy-specific training entry points
├── evaluation/   # Evaluation scripts
├── weights/      # Trained models: <strategy>_seed<N>.pth
├── data/         # Training logs and diagnostics
└── bar_plots/    # Evaluation plots
```

## Training

`training.train.train(name, config, sampler)` trains one DQN agent and saves its best checkpoint (highest average over the last 20 episodes) to `weights/<name>.pth`. It also writes a training log and plot under `data/`.

The pole length for each episode comes from a sampler, when supplied, or from a uniform draw within `Config.length_range` otherwise. A sampler implements `sample()` and `update(length, steps)`.

## Strategy runners

Run these from the project root with the environment active:

```bash
python -m runners.run_baseline
python -m runners.run_domain_randomization
python -m runners.run_boundary_focused
python -m runners.run_failure_focused
```

The baseline uses a fixed pole length of 0.5. Domain randomization samples lengths from 0.4 to 1.8. Boundary-focused and failure-focused runners use their respective samplers.

## Evaluation

Set `STRATEGY_NAME` in `evaluation/test_script.py` to the matching weight prefix, such as `baseline`, `domain_randomized`, `boundary_focused`, or `failure_focused`. The evaluator loads seeds 0-9 and tests 30 pole lengths with ten trials per model and length.

```bash
python -m evaluation.test_script
```
