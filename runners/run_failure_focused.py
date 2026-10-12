from config.settings import Config
from samplers.failure_focused import FailureFocusedSampler, FailureFocusedTrainingLogger
from training.train import train


if __name__ == "__main__":
    for seed in range(10):
        run_name = f"failure_focused_seed{seed}"

        config = Config(seed=seed, episodes=1000)
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
