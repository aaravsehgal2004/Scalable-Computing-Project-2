"""
analysis_t2_sweeps.py — REPORT-ONLY helper (NOT part of the graded zip).

AI-ASSISTANT DECLARATION: Claude (Anthropic) helped draft this script.

Extra OAT T2 sweeps for rate, mean fidelity and p95 latency with 95% CI
(t-interval over seeds), SLO lines, and error-bar figures. Needs matplotlib
only for plotting (falls back to JSON if missing).
"""
import copy, json, sys
from experiments import mean_ci95
from policy import make_policy
from simulator import run_variant_scenario

METRICS = [("rate", "Rate (pairs/s)", "min_rate"),
           ("mean_fidelity", "Mean fidelity", "min_fidelity"),
           ("p95_latency_ms", "p95 latency (ms)", "max_p95_latency_ms")]


def sweep(params, seeds):
    sim_ms = params["grading"]["sim_time_ms"]
    k = params["experiment"]["seeds_per_point"]
    pts = []
    for t2 in params["experiment"]["t2_grid_ms"]:
        p = copy.deepcopy(params)
        p["physics"]["t2_ms"] = t2
        runs = [run_variant_scenario(p, sim_ms, s, make_policy(p))
                for s in seeds[:k]]
        pt = {"t2_ms": t2, "n_seeds": len(runs)}
        for key, _, _ in METRICS:
            m, h = mean_ci95([r[key] for r in runs])
            pt[key + "_mean"], pt[key + "_ci95"] = m, h
        pts.append(pt)
    return pts


if __name__ == "__main__":
    params = json.load(open(sys.argv[1] if len(sys.argv) > 1
                            else "variant_params.json"))
    pts = sweep(params, params["experiment"]["seeds"])
    json.dump(pts, open("t2_sweeps_all_metrics.json", "w"), indent=2)
    for p in pts:
        print(p)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing: JSON written, no figures"); sys.exit()
    xs = [p["t2_ms"] for p in pts]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, (key, lab, slo) in zip(axs, METRICS):
        ax.errorbar(xs, [p[key + "_mean"] for p in pts],
                    yerr=[p[key + "_ci95"] for p in pts],
                    marker="o", capsize=3)
        ax.axhline(params["slo"][slo], ls="--", c="r", label="SLO")
        ax.set_xscale("log"); ax.set_xlabel("T2 (ms)"); ax.set_ylabel(lab)
        ax.legend()
    fig.tight_layout(); fig.savefig("t2_sensitivity.png", dpi=150)
    print("saved t2_sensitivity.png")