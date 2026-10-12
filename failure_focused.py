'''
Strategy 5: Adaptive Failure-Focused Pole-Length Sampling.
'''
import csv
import os
import random


class FailureFocusedSampler:
    '''
    Adaptive Failure-Focused Pole-Length Sampling (Strategy 5).

    Divides the continuous length range into equal-width bins and tracks an EMA
    of survival steps per bin. After a uniform warm-up period, samples bins with
    probability biased toward currently weaker bins, mixed with a uniform
    component so the full training range stays covered.
    '''

    def __init__(
        self,
        length_range=(0.4, 1.8),
        num_bins=10,
        warmup_episodes=100,
        ema_alpha=0.1,
        focus_weight=0.8,
        seed=0,
    ):
        self.low, self.high = length_range
        self.num_bins = num_bins
        self.warmup_episodes = warmup_episodes
        self.ema_alpha = ema_alpha
        self.focus_weight = focus_weight
        # Dedicated RNG so sampler randomness is reproducible without
        # interfering with the global random state used by DQN exploration.
        self.rng = random.Random(seed)

        self.bin_width = (self.high - self.low) / self.num_bins
        self.bin_edges = [
            self.low + i * self.bin_width for i in range(self.num_bins + 1)
        ]

        # Per-bin state. EMA starts as None so unobserved bins are explicit.
        self.ema = [None] * self.num_bins
        self.observations = [0] * self.num_bins
        self.times_sampled = [0] * self.num_bins
        self.samples_taken = 0  # counts sample() calls (= training episodes)

        self._eps = 1e-8

        # Read-only metadata from the most recent sample()/update() for logging.
        # Recording these must not consume RNG or alter sampling behaviour.
        self._last_bin_index = None
        self._last_phase = None
        self._last_selected_probability = None
        self._last_ema_before = None
        self._last_ema_after = None

    def _bin_index(self, length):
        '''Map a continuous length to a bin; upper endpoint belongs to last bin.'''
        if length >= self.high:
            return self.num_bins - 1
        if length <= self.low:
            return 0
        idx = int((length - self.low) / self.bin_width)
        return min(idx, self.num_bins - 1)

    def _sample_in_bin(self, bin_idx):
        '''Continuous uniform length inside the selected bin.'''
        lo = self.bin_edges[bin_idx]
        hi = self.bin_edges[bin_idx + 1]
        return self.rng.uniform(lo, hi)

    def _bin_probabilities(self):
        '''
        Failure-focused distribution mixed with uniform coverage.

        P_i = (1 - focus_weight) / num_bins + focus_weight * normalized_difficulty_i
        '''
        uniform = 1.0 / self.num_bins
        difficulties = self._difficulties()

        total_diff = sum(difficulties)
        if total_diff <= self._eps:
            # Degenerate case: treat as uniform difficulty.
            normalized = [uniform] * self.num_bins
        else:
            normalized = [d / total_diff for d in difficulties]

        probs = [
            (1.0 - self.focus_weight) * uniform + self.focus_weight * p
            for p in normalized
        ]
        # Numerical hygiene: force exact sum of 1 without changing intent.
        total = sum(probs)
        return [p / total for p in probs]

    def _difficulties(self):
        '''
        Min-max difficulty from EMA performance for observed bins.
        Unobserved bins get max difficulty (1.0) so they are not starved,
        without inventing a fake performance score.
        '''
        observed = [e for e in self.ema if e is not None]
        if not observed:
            return [1.0] * self.num_bins

        ema_min = min(observed)
        ema_max = max(observed)

        # All observed EMAs effectively equal -> uniform difficulty.
        if abs(ema_max - ema_min) <= self._eps:
            return [1.0] * self.num_bins

        difficulties = []
        for ema in self.ema:
            if ema is None:
                difficulties.append(1.0)
            else:
                difficulties.append(
                    1.0 - (ema - ema_min) / (ema_max - ema_min + self._eps)
                )
        return difficulties

    def sample(self):
        '''Return the pole length for the next training episode.'''
        if self.samples_taken < self.warmup_episodes:
            # Warm-up: uniform bin choice. Selection prob is exactly 1/num_bins.
            probs = [1.0 / self.num_bins] * self.num_bins
            bin_idx = self.rng.randrange(self.num_bins)
            phase = "warmup"
        else:
            probs = self._bin_probabilities()
            bin_idx = self.rng.choices(range(self.num_bins), weights=probs, k=1)[0]
            phase = "adaptive"

        length = self._sample_in_bin(bin_idx)
        self.times_sampled[bin_idx] += 1
        self.samples_taken += 1

        # Store the probability that actually governed this selection.
        self._last_bin_index = bin_idx
        self._last_phase = phase
        self._last_selected_probability = probs[bin_idx]
        return float(length)

    def update(self, length, steps):
        '''Update the EMA survival estimate for the bin that contained `length`.'''
        bin_idx = self._bin_index(length)
        self._last_ema_before = self.ema[bin_idx]
        self.observations[bin_idx] += 1

        if self.ema[bin_idx] is None:
            # First observation in this bin: initialise EMA to the raw survival.
            self.ema[bin_idx] = float(steps)
        else:
            self.ema[bin_idx] = (
                (1.0 - self.ema_alpha) * self.ema[bin_idx]
                + self.ema_alpha * float(steps)
            )
        self._last_ema_after = self.ema[bin_idx]

    def last_bin_index(self):
        return self._last_bin_index

    def last_phase(self):
        return self._last_phase

    def last_selected_probability(self):
        return self._last_selected_probability

    def last_ema_before(self):
        return self._last_ema_before

    def last_ema_after(self):
        return self._last_ema_after

    def ema_state(self):
        '''Copy of current per-bin EMA values (None = unobserved).'''
        return list(self.ema)

    def next_bin_probabilities(self):
        '''
        Distribution that would be used to choose the NEXT episode.
        Uniform while still in warm-up; adaptive once warm-up is complete.
        Read-only: does not touch the sampler RNG.
        '''
        if self.samples_taken < self.warmup_episodes:
            return [1.0 / self.num_bins] * self.num_bins
        return self._bin_probabilities()

    def save_diagnostics(self, path=None, name="failure_focused_seed0"):
        '''
        Write one CSV row per bin with sampling / performance statistics.
        Unobserved EMA values are written as empty cells (NA).
        '''
        os.makedirs("data", exist_ok=True)
        if path is None:
            path = os.path.join("data", f"{name}_sampler_diagnostics.csv")

        # Final probabilities: after warm-up use adaptive mix; during / if never
        # past warm-up, report the uniform warm-up distribution.
        if self.samples_taken >= self.warmup_episodes:
            probs = self._bin_probabilities()
        else:
            probs = [1.0 / self.num_bins] * self.num_bins

        with open(path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([
                "bin_index",
                "bin_lower",
                "bin_upper",
                "times_sampled",
                "observations",
                "ema_performance",
                "final_sampling_probability",
            ])
            for i in range(self.num_bins):
                ema_value = "" if self.ema[i] is None else f"{self.ema[i]:.6f}"
                writer.writerow([
                    i,
                    f"{self.bin_edges[i]:.6f}",
                    f"{self.bin_edges[i + 1]:.6f}",
                    self.times_sampled[i],
                    self.observations[i],
                    ema_value,
                    f"{probs[i]:.8f}",
                ])
        return path

    def print_summary(self):
        '''Small human-readable dump of per-bin sampling counts and EMA.'''
        print("FailureFocusedSampler summary")
        print(f"  samples_taken={self.samples_taken}  "
              f"warmup={self.warmup_episodes}  "
              f"focus_weight={self.focus_weight}")
        for i in range(self.num_bins):
            ema_str = "NA" if self.ema[i] is None else f"{self.ema[i]:.1f}"
            print(
                f"  bin {i:2d} [{self.bin_edges[i]:.3f}, {self.bin_edges[i + 1]:.3f}]  "
                f"sampled={self.times_sampled[i]:4d}  "
                f"obs={self.observations[i]:4d}  "
                f"ema={ema_str}"
            )


class FailureFocusedTrainingLogger:
    '''
    Episode-by-episode Strategy 5 diagnostics.

    Intended to be passed as train(..., episode_callback=logger).
    Relies only on FailureFocusedSampler read-only metadata; does not sample
    or advance the sampler RNG itself.
    '''

    def __init__(self, sampler, name="failure_focused_seed0"):
        self.sampler = sampler
        os.makedirs("data", exist_ok=True)
        self.path = os.path.join("data", f"{name}_training_diagnostics.csv")

        self._ema_cols = [f"ema_bin_{i}" for i in range(sampler.num_bins)]
        self._prob_cols = [f"prob_bin_{i}" for i in range(sampler.num_bins)]
        self.fieldnames = [
            "episode",
            "phase",
            "pole_length",
            "bin_index",
            "steps_survived",
            "epsilon",
            "ema_before",
            "ema_after",
            "selected_bin_probability",
            *self._ema_cols,
            *self._prob_cols,
        ]

        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
        self._writer.writeheader()
        self._file.flush()

    @staticmethod
    def _fmt_ema(value):
        return "" if value is None else f"{value:.6f}"

    def __call__(self, info):
        '''train() episode_callback: log one completed episode.'''
        sampler = self.sampler
        emas = sampler.ema_state()
        next_probs = sampler.next_bin_probabilities()

        row = {
            "episode": info["episode"],
            "phase": sampler.last_phase(),
            "pole_length": f"{info['pole_length']:.8f}",
            "bin_index": sampler.last_bin_index(),
            "steps_survived": info["steps_survived"],
            "epsilon": f"{info['epsilon']:.6f}",
            "ema_before": self._fmt_ema(sampler.last_ema_before()),
            "ema_after": self._fmt_ema(sampler.last_ema_after()),
            "selected_bin_probability": f"{sampler.last_selected_probability():.8f}",
        }
        for i, ema in enumerate(emas):
            row[f"ema_bin_{i}"] = self._fmt_ema(ema)
        for i, prob in enumerate(next_probs):
            row[f"prob_bin_{i}"] = f"{prob:.8f}"

        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        if not self._file.closed:
            self._file.close()
        return self.path
