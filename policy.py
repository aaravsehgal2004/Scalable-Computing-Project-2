"""
policy.py — QIS-NET Project 2 policy (STUDENT FILE: replace the baseline).

The policy decides, each time a fresh pair arrives on a link, what to do:
swap immediately, purify first, hold, or discard aged pairs. Your variant's
SLOs (rate, fidelity, p95 latency) will typically NOT be met by the
baseline policy below.

Design space (see spec section 3.2):
  * purify while the freshest link fidelity is below a threshold AND a
    second pair is available on the same link;
  * hold vs swap rules (e.g., wait for a purified partner);
  * discard pairs whose current fidelity has fallen below a floor.
"""

import math

# ---- tuning knobs (chosen in T16; see the sweep table in the report) ----
DEFAULT_MAX_LEVEL = 2          # purification rounds on the abundant link (0-3)
DEFAULT_MAX_WAIT_MS = 20.0     # discard abundant-link pairs older than this

class BaselinePolicy:
    """Swap-as-soon-as-possible; never purify. PROVIDED as the baseline.
    It meets the reference scenarios' expectations but is deliberately
    insufficient for most variant SLOs."""

    def __init__(self, params):
        self.params = params
        # map each repeater node to its adjacent link pairs for swapping
        self.chains = params["policy"]["swap_chains"]  # list of [left, right]

    def on_pair_ready(self, link, pair, state, sim):
        for left, right in self.chains:
            if state.pairs.get(left) and state.pairs.get(right):
                sim.swap(left, right)

class FreshestFirstPolicy:
    """Our policy. Same constructor and on_pair_ready hook as the baseline."""

    def __init__(self, params, max_level=DEFAULT_MAX_LEVEL,
                 max_wait_ms=DEFAULT_MAX_WAIT_MS):
        self.params = params
        self.chains = params["policy"]["swap_chains"]
        self.max_level = max_level
        self.max_wait_ms = max_wait_ms
        self.purify_on = params["physics"]["purification_allowed"]

        # Work out which link is scarce: the one with the lowest pair rate
        # (attempt_rate x p_success). Every other link is "abundant".
        links = params["topology"]["links"]
        rates = [l["attempt_rate"] * l["p_success"] for l in links]
        self.scarce = rates.index(min(rates))
        self.abundant = [i for i in range(len(links)) if i != self.scarce]

    # ---------------------------------------------------------- the hook
    def on_pair_ready(self, link, pair, state, sim):
        # A new pair has just arrived on `link`.
        if link.link_id in self.abundant:
            self.discard_stale(link.link_id, state, sim)
            if self.purify_on:
                self.purify_up(link.link_id, state, sim)
        self.try_swaps(state, sim)

    # ---------------------------------------------------------- step 2
    def discard_stale(self, link_id, state, sim):
        """Throw away pairs that have waited longer than max_wait_ms.
        The list is oldest-first, so we keep removing index 0."""
        inv = state.pairs.get(link_id, [])
        while inv and state.now - inv[0].born > self.max_wait_ms:
            sim.discard(link_id, 0)

    # ---------------------------------------------------------- step 3
    def purify_up(self, link_id, state, sim):
        """While two pairs of the same level exist (below max_level),
        purify them into one pair of the next level."""
        inv = state.pairs.get(link_id, [])
        changed = True
        while changed:
            changed = False
            for level in range(self.max_level):
                same = [p for p in inv if level_of(p) == level]
                if len(same) >= 2:
                    # sim.purify() always uses the two pairs at the FRONT of
                    # the list, so move the two freshest same-level pairs there.
                    a, b = same[-1], same[-2]
                    inv.remove(a)
                    inv.remove(b)
                    inv.insert(0, b)
                    inv.insert(0, a)
                    if sim.purify(link_id):
                        inv[-1].level = level + 1   # the new purified pair
                    changed = True
                    break        # start again from level 0

    # ---------------------------------------------------------- step 4
    def try_swaps(self, state, sim):
        """For each repeater: if both sides hold a pair, swap now, using the
        best pair on each side."""
        for left, right in self.chains:
            while state.pairs.get(left) and state.pairs.get(right):
                li = best_index(state.pairs[left], state)
                ri = best_index(state.pairs[right], state)
                sim.swap(left, right, left_index=li, right_index=ri)


# ---------------------------------------------------------- small helpers
def level_of(pair):
    """How many purification rounds made this pair (fresh pairs = 0)."""
    return getattr(pair, "level", 0)


def best_index(pairs, state):
    """Index of the pair with the highest fidelity right now."""
    fids = [state.current_fidelity(p) for p in pairs]
    return fids.index(max(fids))

def make_policy(params):
    """Return your policy object. Replace BaselinePolicy with your own class
    (same constructor signature and on_pair_ready hook)."""
    return FreshestFirstPolicy(params)
