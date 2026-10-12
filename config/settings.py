from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:
    # for training
    length_range: Tuple[float, float] = (0.4, 1.8)  # pole lengths from this range
    episodes: int = 1000                             # episodes to train
    seed: int = 0                                   # seed for reproducible runs

    # for Learning
    gamma: float = 0.99            # how much future reward counts
    learning_rate: float = 5e-4    # learning rate of the optimizer
    batch_size: int = 64           # steps learned from per update
    buffer_size: int = 50_000      # steps the replay buffer remembers
    tau: float = 0.005             # how fast the target network follows

    # for Exploration
    eps_start: float = 1.0         # start fully random
    eps_end: float = 0.05          # end at 5% random
    eps_decay_steps: int = 10_000  # steps to go from start to end