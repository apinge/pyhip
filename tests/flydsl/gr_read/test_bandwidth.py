"""CPU-only byte-accounting checks: python3 test_bandwidth.py."""

import math

if __package__:
    from .bench_bandwidth import bandwidth_gbs, external_byte_counts
else:
    from bench_bandwidth import bandwidth_gbs, external_byte_counts


def main():
    for rows in range(1, 33):
        counts = external_byte_counts(rows)
        assert counts["w_down"] == counts["w_up"] == 6553600
        assert counts["x"] == rows * 20480
        assert counts["y"] == rows * 5120
        assert counts["read"] == 13107200 + 20480 * rows
        assert counts["io"] == 13107200 + 25600 * rows
        latency_us = 30.0
        cuda_perf_formula = counts["read"] * 1e-6 / (latency_us / 1000)
        assert math.isclose(bandwidth_gbs(counts["read"], latency_us), cuda_perf_formula)
    assert bandwidth_gbs(10_000_000, 10) == 1000
    for latency in (0, -1, float("nan"), float("inf")):
        try:
            bandwidth_gbs(1, latency)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid latency accepted")
    for rows in (0, 33):
        try:
            external_byte_counts(rows)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid rows accepted")
    print("PASS: T=1..32 byte counts, cudaPerf unit conversion, and invalid inputs")


if __name__ == "__main__":
    main()
