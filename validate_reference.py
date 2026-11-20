"""
validate_reference.py — T12 (reference validation) and T13 (BBPSSW ladder).

Usage: python validate_reference.py variant_params.json [--n-seeds 20]
                                    [--out validation_reference_results.json]

AI-assistant declaration: drafted with help from Claude (Anthropic), then
read, run and checked by the group member who owns it (Aarav).
"""

import argparse
import json
import math

import numpy as np

import qnetsim
from simulator import make_links, run_one_hop, run_two_hop

# two-sided 97.5% t quantiles for df = 1..30 (df > 30 uses 1.96 + small corr.)
_T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262,
         2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101,
         2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052,
         2.048, 2.045, 2.042]


def t_crit(df):
    if df <= 30:
        return _T975[df - 1]
    return 1.96 + 2.4 / df          # good to ~3 decimals for df > 30


def mean_ci(xs):
    xs = np.asarray(xs, dtype=float)
    n = len(xs)
    m = float(xs.mean())
    se = float(xs.std(ddof=1) / math.sqrt(n))
    h = t_crit(n - 1) * se
    return m, h


def poisson_tail_sum(l1, l2):
    """E[min(N1,N2)] = sum_k P(N1>=k) P(N2>=k), N_i ~ Poisson(l_i)."""
    kmax = int(min(l1, l2) + 12 * math.sqrt(min(l1, l2)) + 50)
    ks = np.arange(0, kmax + 1)
    lg = np.array([math.lgamma(k + 1.0) for k in ks])

    def tails(lam):
        # tails(lam)[j] = P(N >= j+1); pmf from logs to avoid exp underflow
        pmf = np.exp(-lam + ks * math.log(lam) - lg)
        return 1.0 - np.cumsum(pmf)

    a, b = tails(l1), tails(l2)
    return float(np.sum(np.clip(a, 0, 1) * np.clip(b, 0, 1)))


def row(name, vals, analytic):
    m, h = mean_ci(vals)
    return {"quantity": name, "mean": m, "ci95_half": h,
            "ci_low": m - h, "ci_high": m + h, "analytic": analytic,
            "within_ci": bool(m - h <= analytic <= m + h)}


# ------------------------------------------------------------------ T13
class _LadderPolicy:
    """Two-round BBPSSW ladder on one link (level 0 -> 1 -> 2)."""

    def __init__(self, rounds):
        self.rounds = rounds
        self.pool = [[] for _ in range(rounds + 1)]
        self.raw = {}                      # id(pair) -> raw pairs inside it
        self.att = [0] * rounds
        self.ok = [0] * rounds
        self.fid = [[] for _ in range(rounds + 1)]
        self.outputs = []

    def on_pair_ready(self, link, pair, state, sim):
        state.pairs[link.link_id].pop()
        self.raw[id(pair)] = 1
        self.pool[0].append(pair)
        self.fid[0].append(pair.f0)
        for lv in range(self.rounds):
            while len(self.pool[lv]) >= 2:
                a, b = self.pool[lv].pop(0), self.pool[lv].pop(0)
                tmp = 100 + lv
                state.pairs.setdefault(tmp, []).extend([a, b])
                self.att[lv] += 1
                r = self.raw.pop(id(a)) + self.raw.pop(id(b))
                if sim.purify(tmp):
                    out = state.pairs[tmp].pop()
                    self.ok[lv] += 1
                    self.raw[id(out)] = r
                    self.pool[lv + 1].append(out)
                    self.fid[lv + 1].append(out.f0)
        # pairs reaching the top level are the ladder's outputs
        while self.pool[self.rounds]:
            self.outputs.append(self.pool[self.rounds].pop())


def run_ladder(params, seed, sim_time_ms, rounds=2):
    link = make_links(params, "one_hop")
    sim = qnetsim.Simulator(link, t2=math.inf, q_swap=1.0, f_swap=1.0,
                            purification_allowed=True, seed=seed)
    pol = _LadderPolicy(rounds)
    sim.run(sim_time_ms, pol)
    gen = sim.state.stats["generated"]
    left = sum(1 for lv in range(rounds) for _ in pol.pool[lv])  # approx
    return pol, gen, sim


def t13(params, n_seeds, seeds):
    f0 = params["physics"]["f_link"]
    f1, p1 = qnetsim.bbpssw(f0)
    f2, p2 = qnetsim.bbpssw(f1)
    analytic = {"f0": f0, "f1": f1, "f2": f2, "p1": p1, "p2": p2,
                "raw_per_f1": 2.0 / p1,
                "raw_per_f2": 4.0 / (p1 * p2)}
    sims = {k: [] for k in ("f1", "f2", "p1", "p2", "raw_per_f1", "raw_per_f2")}
    for s in seeds:
        pol, gen, sim = run_ladder(params, s, 120000.0)
        lvl_fid = [np.mean(x) if x else float("nan") for x in pol.fid]
        sims["f1"].append(float(lvl_fid[1]))
        sims["f2"].append(float(lvl_fid[2]))
        sims["p1"].append(pol.ok[0] / pol.att[0])
        sims["p2"].append(pol.ok[1] / pol.att[1])
        # raw consumed = raws of all pairs not still waiting in the pools
        waiting = sum(pol.raw[id(p)] for lv in pol.pool for p in lv)
        consumed = gen - waiting
        sims["raw_per_f1"].append(2.0 * pol.att[0] / pol.ok[0])
        sims["raw_per_f2"].append(consumed / len(pol.outputs))
    rows = [row(k, sims[k], analytic[k]) for k in
            ("f1", "f2", "p1", "p2", "raw_per_f1", "raw_per_f2")]
    return analytic, rows


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("params")
    ap.add_argument("--n-seeds", type=int, default=20)
    ap.add_argument("--out", default="validation_reference_results.json")
    args = ap.parse_args()

    with open(args.params) as fh:
        params = json.load(fh)
    T = float(params["grading"]["sim_time_ms"])
    exp = params["reference"]["expected"]
    # validation seeds are deliberately NOT the grading seeds
    seeds = [9100 + 17 * i for i in range(args.n_seeds)]

    one = [run_one_hop(params, T, s) for s in seeds]
    two = [run_two_hop(params, T, s) for s in seeds]

    r1 = params["reference"]["one_hop"]
    lam1 = r1["attempt_rate"] * r1["p_success"]
    l2 = params["reference"]["two_hop"]["links"]
    la = l2[0]["attempt_rate"] * l2[0]["p_success"]
    lb = l2[1]["attempt_rate"] * l2[1]["p_success"]
    f_link = params["physics"]["f_link"]
    f_swap = params["physics"]["f_swap"]

    exact_two_rate = poisson_tail_sum(la * T, lb * T) / (T / 1000.0)
    asym_two_rate = min(la, lb) * 1000.0

    rows = [
        row("one_hop rate (/s)", [r["rate"] for r in one], lam1 * 1000.0),
        row("one_hop mean_fidelity", [r["mean_fidelity"] for r in one], f_link),
        row("two_hop rate (/s) vs exact finite-time",
            [r["rate"] for r in two], exact_two_rate),
        row("two_hop mean_fidelity", [r["mean_fidelity"] for r in two],
            f_link ** 2 * f_swap),
    ]

    print(f"{args.n_seeds} seeds, T = {T:.0f} ms")
    print(f"tracker values in json : {exp}")
    print(f"two-hop asymptotic rate {asym_two_rate:.3f}/s, exact finite-time "
          f"{exact_two_rate:.3f}/s")
    print(f"{'quantity':45s} {'mean':>9s} {'+/-95%':>8s} {'analytic':>9s}  ok")
    for r in rows:
        print(f"{r['quantity']:45s} {r['mean']:9.4f} {r['ci95_half']:8.4f} "
              f"{r['analytic']:9.4f}  {'YES' if r['within_ci'] else 'NO'}")

    analytic13, rows13 = t13(params, args.n_seeds, seeds)
    print("\nT13 BBPSSW ladder: f0={f0:.4f} -> f1={f1:.4f} -> f2={f2:.4f}; "
          "p1={p1:.4f}, p2={p2:.4f}".format(**analytic13))
    for r in rows13:
        print(f"{r['quantity']:45s} {r['mean']:9.4f} {r['ci95_half']:8.4f} "
              f"{r['analytic']:9.4f}  {'YES' if r['within_ci'] else 'NO'}")

    # determinism check
    det = (run_one_hop(params, T, 1002) == run_one_hop(params, T, 1002)
           and run_two_hop(params, T, 1002) == run_two_hop(params, T, 1002))
    print("\ndeterministic for same seed:", det)

    with open(args.out, "w") as fh:
        json.dump({"n_seeds": args.n_seeds, "seeds": seeds, "sim_time_ms": T,
                   "tracker_expected": exp,
                   "two_hop_rate_asymptotic": asym_two_rate,
                   "two_hop_rate_exact_finite_time": exact_two_rate,
                   "T12": rows, "T13_analytic": analytic13, "T13": rows13,
                   "deterministic": det}, fh, indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
