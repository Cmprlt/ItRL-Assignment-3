from boundary_focused_sampler import BoundarySampler
from train import train
from config import Config

if __name__ == "__main__":
    for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        sampler = BoundarySampler()
        train(f"boundary_focused_seed{seed}",Config(seed = seed,episodes = 1000), sampler= sampler)

