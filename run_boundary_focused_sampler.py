from boundary_focused_sampler import BoundarySampler
from train import train
from config import Config

if __name__ == "__main__":
    sampler = BoundarySampler()
    train("boundry_focused_seed0_1000ep", Config(episodes = 1000), sampler= sampler)


