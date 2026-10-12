# ItRL-Assignment-3

DQN agents for CartPole that are trained to balance poles of different lengths.
Every strategy changes only which pole length is used in each training episode.
The network, the DQN algorithm and the hyperparameters stay the same, so results can be compared directly. All strategies run on seeds 0-9, 1000 episodes per seed.

## Setup
To set up the environment and install requirements run:
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Project structure

```
├── config.py                        # hyperparameters and default length range (0.4–1.8)
├── network.py                       # QNetwork (the standard network from the test script)
├── train.py                         # DQN training loop + baseline runs
├── boundary_focused_sampler.py      # BoundarySampler (boundary-focused strategy)
├── run_boundary_focused_sampler.py  # trains the boundary-focused models
├── test_script.py                   # evaluation over the 30 official pole lengths
├── weights/                         # trained models: <strategy>_seed<N>.pth
└── data/                            # training logs and plots per run
```

## How training works

`train(name, config, sampler)` in `train.py` trains one DQN agent and saves the best checkpoint
(highest average over the last 20 episodes) to `weights/<name>.pth`.
It also writes `data/<name>_training_log.csv` and `data/<name>_training_plot.png`.

The pole length for each episode comes from:
- a **sampler**, if one is passed, or
- `random.uniform(*config.length_range)` otherwise.

A sampler is any class with two methods:

```python
sample()               # returns the pole length for the next episode
update(length, steps)  # receives how many steps the agent survived at that length
```

## Strategies

### Baseline

Trained on a single pole length of 0.5 (the default CartPole length).

To train run:
```
python train.py
```

### Boundary-focused pole sampling

Trains only on a few fixed **anchor lengths**, with the two extremes trained twice as often:

| Length | 0.4 | 0.75 | 1.1 | 1.45 | 1.8 |
|---|---|---|---|---|---|
| Weight | 2 | 1 | 1 | 1 | 2 |

Lengths are drawn without repeats in cycles of 7 episodes, so every cycle has exactly this distribution.
This tests whether training on diverse anchor environments lets the Q-network generalize to unseen lengths.

To train run:
```
python run_boundary_focused_sampler.py
```
