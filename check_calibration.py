"""
check_calibration.py — T9: confirm run_variant_scenario + BaselinePolicy
reproduces the calibration block in variant_params.json
(params['calibration']['baseline_policy'], at grading seed / sim time).

Usage:  python check_calibration.py [variant_params.json]
Exit code 0 if every metric matches within tolerance, 1 otherwise.
"""

import json
import sys

from policy import BaselinePolicy
from simulator import run_variant_scenario

METRICS = ["rate", "mean_fidelity", "p95_latency_ms",
           "pairs_consumed_per_delivery"]
REL_TOL = 1e-3   # calibration values are rounded to 4 d.p.


def main(path):
    params = json.load(open(path))
    g = params["grading"]
    expected = params["calibration"]["baseline_policy"]
    got = run_variant_scenario(params, g["sim_time_ms"], g["seed"],
                               BaselinePolicy(params))

    ok_all = True
    print(f"seed={g['seed']}  sim_time_ms={g['sim_time_ms']}")
    print(f"{'metric':30s} {'expected':>12s} {'got':>12s}  status")
    for k in METRICS:
        e, v = expected[k], got[k]
        ok = abs(v - e) <= REL_TOL * max(1.0, abs(e))
        ok_all &= ok
        print(f"{k:30s} {e:12.4f} {v:12.4f}  {'OK' if ok else 'MISMATCH'}")
    print("PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "variant_params.json"))
