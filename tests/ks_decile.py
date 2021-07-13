"""Kolmogorov–Smirnov test to assert goodness of a binary model."""
import numpy as np


def separate_in_bins(x, num_bins: int = 10):
    bins = np.quantile(x, [0.1 * i for i in reversed(range(10 + 1))])
    bins[-1] -= 1
    bin_inds = np.digitize(x, bins, right=True) - 1
    return bin_inds


def count_bin_events(y, bin_inds, num_bins: int = 10):
    bin_inds_used, bin_freqs = np.unique(bin_inds, return_counts=True)

    rate_pos, rate_neg = np.zeros((2, num_bins), dtype=float)
    np.add.at(rate_pos, bin_inds, y)

    rate_neg[bin_inds_used] = (bin_freqs - rate_pos[bin_inds_used]) / float(
        sum(1.0 - y)
    )
    rate_pos[bin_inds_used] /= float(sum(y))

    return rate_pos, rate_neg


def ks_test(x, y, num_bins: int = 10, percentage: bool = True):
    bin_inds = separate_in_bins(x, num_bins=num_bins)
    rate_pos, rate_neg = count_bin_events(y, bin_inds, num_bins=num_bins)

    cdf_pos = np.cumsum(rate_pos)
    cdf_neg = np.cumsum(rate_neg)

    diffs = np.abs(cdf_pos - cdf_neg)

    max_ind = np.argmax(diffs)
    ks_stat = float(diffs[max_ind]) * (100 ** int(percentage))

    return ks_stat, max_ind


def _test():
    import scipy.stats

    n = 400
    np.random.seed(16)
    y = np.random.randint(2, size=n)
    x = np.clip(y * (0.90 - 0.10) + 0.10 + 0.5 * np.random.randn(n), 0.0, 1.0)
    noise_inds = np.random.choice(n, size=int(n * 0.2), replace=False)
    x[noise_inds] = 1.0 - x[noise_inds]

    ks_stat, max_ind = ks_test(x=x, y=y)
    print(ks_stat, max_ind)
    print(
        "Verdict: (possibly)",
        "Good model." if ks_stat > 40.0 and max_ind < 3 else "Bad model.",
    )

    print("Comparing to scipy KS test:")
    ks_stat, p_val = scipy.stats.ks_2samp(x >= 0.5, x < 0.5)
    print("(scipy) KS stat:", ks_stat)
    print("p_val:", p_val)
    print("Null hypothesis: the distributions are the same.")


if __name__ == "__main__":
    _test()
