import argparse

from config import Config
from failure_focused import FailureFocusedSampler, FailureFocusedTrainingLogger
from train import train


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Strategy 5: Adaptive Failure-Focused Pole-Length Sampling."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for Config and FailureFocusedSampler (default: 0).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seed = args.seed
    run_name = f"failure_focused_seed{seed}"

    config = Config(seed=seed)
    sampler = FailureFocusedSampler(
        length_range=config.length_range,
        seed=seed,
    )
    logger = FailureFocusedTrainingLogger(sampler, name=run_name)

    try:
        train(
            run_name,
            config=config,
            sampler=sampler,
            episode_callback=logger,
        )
    finally:
        training_diagnostics_path = logger.close()

    diagnostics_path = sampler.save_diagnostics(name=run_name)
    sampler.print_summary()
    print(f"Saved sampler diagnostics to {diagnostics_path}")
    print(f"Saved training diagnostics to {training_diagnostics_path}")
