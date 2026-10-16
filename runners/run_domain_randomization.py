from config.settings import Config
from training.train import train


if __name__ == "__main__":
    for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        train(f"domain_randomized_seed{seed}", Config(length_range=(0.4, 1.8), episodes=1000, seed=seed))