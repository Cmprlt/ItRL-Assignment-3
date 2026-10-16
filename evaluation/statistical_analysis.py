"""Test whether the strategies differ significantly with `python -m evaluation.statistical_analysis`.

Each trained model (strategy x seed) is evaluated with the same protocol as test_script.py
(30 pole lengths x 10 trials) and summarised by its mean score. One seed is one independent
sample, so n = 10 per strategy. Every pair of strategies is compared with Welch's t-test
(two-sided, no equal-variance assumption, as recommended by Colas et al., 2019).
"""
import os
from itertools import combinations
from multiprocessing import Pool

import gym
import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_ind

from evaluation.test_script import test_pole_length
from models.q_network import QNetwork

STRATEGIES = ["baseline", "domain_randomized", "boundary_focused", "failure_focused"]
SEEDS = range(10)
POLE_LENGTHS = np.linspace(0.4, 1.8, 30)
RUNS_PER_MODEL = 10
RESULTS_PATH = "data/seed_scores.csv"
SUMMARY_PATH = "data/strategy_summary.csv"
TESTS_PATH = "data/welch_tests.csv"


def seed_score(job):
    """Mean test score of one trained model over all pole lengths and trials."""
    strategy, seed = job
    torch.set_num_threads(1)  # models are evaluated in parallel processes
    model = QNetwork(4, 2)
    model.load_state_dict(torch.load(f"weights/{strategy}_seed{seed}.pth", weights_only=True))
    model.eval()

    scores = []
    for length in POLE_LENGTHS:
        for _ in range(RUNS_PER_MODEL):
            env = gym.make("CartPole-v1")
            env.unwrapped.length = length
            with torch.no_grad():
                scores.append(test_pole_length(env, model))
            env.close()
    mean_score = np.mean(scores)
    return {"strategy": strategy, "seed": seed, "score": mean_score}


if __name__ == "__main__":
    # Evaluating all 40 models
    already_evaluated = os.path.exists(RESULTS_PATH)
    if already_evaluated == False:
        jobs = []
        for strategy in STRATEGIES:
            for seed in SEEDS:
                jobs.append((strategy, seed))
        with Pool() as pool:
            rows = pool.map(seed_score, jobs)
        pd.DataFrame(rows).to_csv(RESULTS_PATH, index=False)

    df = pd.read_csv(RESULTS_PATH)

    # Per-seed scores of each strategy
    scores = {}
    for strategy in STRATEGIES:
        scores[strategy] = df.loc[df.strategy == strategy, "score"].to_numpy()

    # Mean and std of the per-seed scores of each strategy
    summary_rows = []
    for strategy in STRATEGIES:
        mean = scores[strategy].mean()
        std = scores[strategy].std(ddof=1)
        summary_rows.append({"strategy": strategy, "mean": mean, "std": std})
    pd.DataFrame(summary_rows).to_csv(SUMMARY_PATH, index=False)

    # Welch's t-test (alpha = 0.05) for every pair of strategies
    test_rows = []
    for a, b in combinations(STRATEGIES, 2):
        t, p = ttest_ind(scores[a], scores[b], equal_var=False)

        # Cohen's d: difference in means divided by the pooled standard deviation
        mean_a = scores[a].mean()
        mean_b = scores[b].mean()
        var_a = scores[a].var(ddof=1)
        var_b = scores[b].var(ddof=1)
        pooled_std = np.sqrt((var_a + var_b) / 2)
        d = (mean_a - mean_b) / pooled_std

        significant = False
        if p < 0.05:
            significant = True
        test_rows.append({"strategy_a": a, "strategy_b": b, "t": t, "p": p,
                          "cohens_d": d, "significant": significant})
    pd.DataFrame(test_rows).to_csv(TESTS_PATH, index=False)
