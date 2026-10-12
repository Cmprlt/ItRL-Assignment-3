from config.settings import Config
from samplers.test_sampler import TestSampler
from training.train import train

if __name__ == "__main__":
    train("sampler_test", Config(episodes = 5), sampler= TestSampler())
