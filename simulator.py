"""
simulator.py — QIS-NET Project 2 runner (STUDENT FILE: complete the TODOs).

Drive the provided harness (qnetsim.py) to implement the three graded
entry points. Signatures are normative; the autograder imports them.

Only numpy + standard library + qnetsim may be used.
AI-assistant declaration:  run_variant_scenario and  _fifo_latencies and _p95  and other test files 
were drafted and modified with help from Claude and I read, tested and understood the code (Bangchen).
"""

import math

import numpy as np

from qnetsim import Link, Simulator, summarise


# ---------------------------------------------------------------- helpers


def make_links(params, which="variant"):
    """Build Link objects from the variant parameter dict.

    `which="one_hop"` / `"two_hop"` select the reference scenarios defined in
    params["reference"]; `which="variant"` builds the full variant topology.
    PROVIDED — you should not need to change this.
    """
    links = []
    if which == "one_hop":
        ref = params["reference"]["one_hop"]
        links.append(Link(link_id=0, nodes=(0, 1),
                          attempt_rate=ref["attempt_rate"],
                          p_success=ref["p_success"],
                          f_link=params["physics"]["f_link"]))
    elif which == "two_hop":
        ref = params["reference"]["two_hop"]
        for i, lp in enumerate(ref["links"]):
            links.append(Link(link_id=i, nodes=(i, i + 1),
                              attempt_rate=lp["attempt_rate"],
                              p_success=lp["p_success"],
                              f_link=params["physics"]["f_link"]))
    else:
        for i, lp in enumerate(params["topology"]["links"]):
            links.append(Link(link_id=i, nodes=tuple(lp["nodes"]),
                              attempt_rate=lp["attempt_rate"],
                              p_success=lp["p_success"],
                              f_link=params["physics"]["f_link"]))
    return links


def request_times(params, sim_time_ms, seed):
    """Deterministic request arrival times (ms) for the variant workload.
    Uses a generator derived from `seed` so runs are reproducible.
    PROVIDED."""
    wl = params["workload"]
    rng = np.random.default_rng(seed + 7919)
    t, times = 0.0, []
    while True:
        t += rng.exponential(1000.0 / wl["arrival_rate_per_s"])
        if t > sim_time_ms:
            break
        times.append(t)
    return times


# ---------------------------------------------------------------- graded entry points


def run_one_hop(params, sim_time_ms, seed) -> dict:
    """Reference scenario: a single elementary link, no swapping.
    Return {"rate": ..., "mean_fidelity": ...}.

    TODO: build the one-hop links, run the Simulator for sim_time_ms with
    policy=None, and use qnetsim.summarise on a delivered-list you fill
    yourself (every generated pair counts as delivered here — record
    deliveries via a tiny inline policy or by reading state.pairs/stats).

    HINT: the harness only records `delivered` on swap; for the one-hop
    case, write a 3-line policy class whose on_pair_ready appends a
    DeliveredPair-equivalent record (or count pairs via state.stats and
    track fidelities in a list).
    """
    raise NotImplementedError("TODO: implement run_one_hop")


def run_two_hop(params, sim_time_ms, seed) -> dict:
    """Reference scenario: two links, one repeater, immediate swap,
    T2 = infinity, q_swap = 1 (params['reference']['two_hop']).
    Return {"rate": ..., "mean_fidelity": ...}.

    TODO: build links, construct the Simulator with t2=math.inf,
    q_swap=1.0, f_swap=params['physics']['f_swap']; use a policy that swaps
    as soon as both links hold a pair; summarise with qnetsim.summarise.
    """
    raise NotImplementedError("TODO: implement run_two_hop")


def run_variant_scenario(params, sim_time_ms, seed, policy) -> dict:
    """Full variant scenario with YOUR policy.
    Return {"rate": ..., "mean_fidelity": ..., "p95_latency_ms": ...,
            "pairs_consumed_per_delivery": ...}.

    TODO:
      * build the variant topology links;
      * construct the Simulator with your variant's physics
        (t2, q_swap, f_swap, purification_allowed, seed);
      * generate request_times(...); use a pooled unlimited end-to-end buffer.
        A request consumes a buffered pair at zero latency, or queues FIFO
        for a later delivery. Exclude unserved requests from p95. Fidelity
        is measured at generation; this buffer does not model further decay;
      * return the four metrics. pairs_consumed_per_delivery =
        stats['generated'] / max(1, delivered).
    """
    phys = params["physics"]
    links = make_links(params, "variant")
    sim = Simulator(
        links=links,
        t2=phys["t2_ms"],
        q_swap=phys["q_swap"],
        f_swap=phys["f_swap"],
        purification_allowed=phys["purification_allowed"],
        seed=seed,
    )
    delivered = sim.run(sim_time_ms, policy=policy)
    summary = summarise(delivered, sim_time_ms)

    reqs = request_times(params, sim_time_ms, seed)
    latencies = _fifo_latencies(reqs, [d.delivered_at for d in delivered])

    return {
        "rate": float(summary["rate"]),
        "mean_fidelity": float(summary["mean_fidelity"]),
        "p95_latency_ms": _p95(latencies),
        "pairs_consumed_per_delivery":
            sim.state.stats["generated"] / max(1, len(delivered)),
    }


# ---------------------------------------------------------------- request matching


def _fifo_latencies(req_times, delivery_times):
    """Match requests to end-to-end pairs through a pooled, unlimited buffer.

    Events are processed in time order. A request that finds a buffered pair
    is served at zero latency; otherwise it joins a FIFO queue and is served
    by the next delivery, with latency = delivery time - arrival time.
    Requests still queued at the end of the run are unserved and excluded.
    On a time tie the delivery is processed first (the pair is ready).
    """
    reqs = sorted(req_times)
    dels = sorted(delivery_times)
    latencies = []
    waiting = []          # FIFO of arrival times of queued requests
    head = 0              # index of the oldest still-waiting request
    buffered = 0          # end-to-end pairs in the pool
    i = j = 0
    while i < len(reqs) or j < len(dels):
        if j < len(dels) and (i >= len(reqs) or dels[j] <= reqs[i]):
            t = dels[j]
            j += 1
            if head < len(waiting):
                latencies.append(t - waiting[head])
                head += 1
            else:
                buffered += 1
        else:
            t = reqs[i]
            i += 1
            if buffered > 0:
                buffered -= 1
                latencies.append(0.0)
            else:
                waiting.append(t)
    return latencies


def _p95(values):
    """95th percentile, nearest-rank (same convention as qnetsim.summarise).
    Returns inf when no request was served."""
    if not values:
        return float("inf")
    s = sorted(values)
    return float(s[min(len(s) - 1, int(math.ceil(0.95 * len(s))) - 1)])
