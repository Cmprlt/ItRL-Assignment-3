from samplers import TestSampler
from train import train
from config import Config

if __name__ == "__main__":
    train("sampler_test", Config(episodes = 5), sampler= TestSampler())
