# ItRL-Assignment-3

## Project layout

- `config/`: training configuration.
- `models/`: Q-network architecture.
- `samplers/`: pole-length sampling strategies and diagnostics.
- `training/`: shared DQN training implementation.
- `runners/`: strategy-specific training entry points.
- `evaluation/`: evaluation scripts.
- `data/`, `weights/`, `bar_plots/`: generated artifacts.

Run commands from the project root with the project environment activated:

```bash
python -m runners.run_baseline
python -m runners.run_domain_randomization
python -m runners.run_boundary_focused
python -m runners.run_failure_focused
python -m evaluation.test_script
```