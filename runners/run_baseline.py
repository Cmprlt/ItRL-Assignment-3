from config.settings import Config
from training.train import train


if __name__ == "__main__":
    for seed in range(10):
        config = Config(length_range=(0.5, 0.5), episodes=1000, seed=seed)
        train(f"baseline_seed{seed}", config)