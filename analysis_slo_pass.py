"""
analysis_slo_pass.py — T15 (REPORT-ONLY helper, not part of the graded zip).

Run the variant scenario over N seeds (default 100) with the current
policy (policy.make_policy) and estimate, for each SLO, the probability
that a single run passes it:

    rate           >= slo.min_rate
    mean_fidelity  >= slo.min_fidelity
    p95_latency_ms <= slo.max_p95_latency_ms
    all three jointly

Each probability is reported with a 95% Wilson score interval. Per-seed
results go to slo_pass_runs.json, the summary to slo_pass_summary.json.

Usage:  python analysis_slo_pass.py [n_seeds] [first_seed] [--baseline]
        --baseline  use policy.BaselinePolicy instead of make_policy
"""

import json
import math
import sys

import numpy as np

from policy import BaselinePolicy, make_policy
from simulator import run_variant_scenario


def wilson(k, n, z=1.96):
    """95% Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def main(argv):
    flags = {a for a in argv if a.startswith("--")}
    pos = [a for a in argv if not a.startswith("--")]
    n = int(pos[0]) if pos else 100
    first = int(pos[1]) if len(pos) > 1 else 10000
    use_baseline = "--baseline" in flags

    with open("variant_params.json") as f:
        params = json.load(f)
    slo = params["slo"]
    sim_ms = params["grading"]["sim_time_ms"]
    checks = {
        "rate": lambda r: r["rate"] >= slo["min_rate"],
        "mean_fidelity": lambda r: r["mean_fidelity"] >= slo["min_fidelity"],
        "p95_latency_ms":
            lambda r: r["p95_latency_ms"] <= slo["max_p95_latency_ms"],
    }

    runs = []
    for seed in range(first, first + n):
        pol = BaselinePolicy(params) if use_baseline else make_policy(params)
        r = run_variant_scenario(params, sim_ms, seed, pol)
        r["seed"] = seed
        r["pass"] = {k: bool(c(r)) for k, c in checks.items()}
        r["pass"]["all"] = all(r["pass"].values())
        runs.append(r)

    summary = {"policy": "baseline" if use_baseline else "make_policy",
               "n_seeds": n, "seeds": [first, first + n - 1],
               "sim_time_ms": sim_ms, "slo": slo, "metrics": {}}
    for k in list(checks) + ["all"]:
        passed = sum(r["pass"][k] for r in runs)
        lo, hi = wilson(passed, n)
        entry = {"passed": passed, "p_pass": passed / n,
                 "wilson95": [lo, hi]}
        if k != "all":
            v = np.array([r[k] for r in runs], dtype=float)
            entry.update(mean=float(v.mean()), std=float(v.std(ddof=1)),
                         p05=float(np.percentile(v, 5)),
                         median=float(np.median(v)),
                         p95=float(np.percentile(v, 95)))
        summary["metrics"][k] = entry

    with open("slo_pass_runs.json", "w") as f:
        json.dump(runs, f, indent=2)
    with open("slo_pass_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"policy={summary['policy']}  seeds {first}..{first + n - 1}")
    print(f"{'SLO':16s} {'pass':>9s} {'P(pass)':>8s}  95% Wilson CI"
          f"   {'mean':>9s} {'median':>9s}")
    for k, e in summary["metrics"].items():
        line = (f"{k:16s} {e['passed']:4d}/{n:<4d} {e['p_pass']:8.2f}"
                f"  [{e['wilson95'][0]:.2f}, {e['wilson95'][1]:.2f}]")
        if "mean" in e:
            line += f"   {e['mean']:9.4f} {e['median']:9.4f}"
        print(line)


if __name__ == "__main__":
    main(sys.argv[1:])
