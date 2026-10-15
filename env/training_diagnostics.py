"""Passive scalar summaries from existing training forwards; never policy inputs."""
import math

import numpy as np


class Summary:
    def __init__(self):
        self.count = self.nonzero = self.positive = self.negative = 0
        self.mean = self.mean_square = 0.0
        self.minimum = self.maximum = None

    def add(self, value):
        value = float(value)
        self.count += 1
        self.nonzero += value != 0
        self.positive += value > 0
        self.negative += value < 0
        self.mean = self.mean * ((self.count - 1) / self.count) + value / self.count
        self.mean_square = self.mean_square * ((self.count - 1) / self.count) + value * value / self.count
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)

    def report(self):
        variance = self.mean_square - self.mean * self.mean
        return {'count': self.count, 'nonzero_count': self.nonzero,
                'positive_count': self.positive, 'negative_count': self.negative,
                'mean': self.mean if self.count else None,
                'std': math.sqrt(max(0.0, variance)) if self.count and math.isfinite(variance) else None,
                'min': self.minimum, 'max': self.maximum}


class EpochDiagnostics:
    def __init__(self, seats, clip=None):
        self.seats = seats
        self.clip = clip
        self.groups = {seat: {'advantage': Summary(), 'entropy': Summary(),
                            'sampled_approx_kl': Summary(), 'ratio_outside_clip_count': 0,
                            'surrogate_clipped_count': 0}
                       for seat in ('ALL', 'BLUE', 'RED', 'UNKNOWN')}

    def observe(self, index, advantage, probs, log_probs, log_ratio=None, clipped=False):
        # Scalars only. Inputs are read without mutating arrays or consuming RNG.
        entropy = float(-np.sum(probs * log_probs))
        outside = (log_ratio is not None and
                   (log_ratio < math.log1p(-self.clip) or log_ratio > math.log1p(self.clip)))
        for seat in ('ALL', self.seats[index]):
            group = self.groups[seat]
            group['advantage'].add(advantage)
            group['entropy'].add(entropy)
            if log_ratio is not None:
                # k1 = E_sample[log pi_old(a) - log pi_current(a)].
                # It may be negative on a finite sampled batch; this is NOT exact KL.
                group['sampled_approx_kl'].add(-log_ratio)
                group['ratio_outside_clip_count'] += bool(outside)
                group['surrogate_clipped_count'] += bool(clipped)

    def report(self):
        result = {}
        for seat, group in self.groups.items():
            count = group['sampled_approx_kl'].count
            result[seat] = {key: value.report() if isinstance(value, Summary) else value
                            for key, value in group.items()}
            result[seat]['sampled_ratio_clip_fraction'] = group['ratio_outside_clip_count'] / count if count else None
            result[seat]['surrogate_clip_fraction'] = group['surrogate_clipped_count'] / count if count else None
        return result
