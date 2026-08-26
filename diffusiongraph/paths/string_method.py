"""
Path type 4 (SEED §3.2): entropy-balanced / finite-temperature string method
(Moreau et al., arXiv:2602.22122).

DEFERRED for the Phase 1 gate -- not a shortcut, a scoped decision, per
Strategic_Blind_Spots_Analysis.md #4: a full finite-temperature string
method requires spawning stochastic walkers per image and simulating SDEs at
fine time discretization, re-parametrizing the string repeatedly. Running
that exhaustively over 45 CIFAR-10 pairs x 3 seeds x 3 sigma_tau levels is
not tractable as a first pass, especially on a 4GB GPU.

No code was released for Moreau et al. either (checked directly; see
THIRD_PARTY.md), so this would also be an from-scratch implementation.

Plan: path type 3 (tangential_geodesic.py) serves as the primary
geometry-aware path and fast proxy for identifying *candidate* routing
edges. Once Phase 1 has a routing matrix, THIS module gets implemented and
run only on the small number of flagged high-C(A,B) pairs, as a
confirmatory check that the routing survives entropy-balancing (guards
against path 3's tangential-only objective producing a route that a fuller
finite-temperature treatment would not).

`enabled_paths` in config.py deliberately omits "string_method" until then.
"""
from __future__ import annotations

from diffusiongraph.paths.base import PathConstructor, PathResult


class StringMethodPath(PathConstructor):
    name = "string_method"

    def construct(self, denoiser, x_a, x_b, class_a, class_b, sigma_tau, n_steps, seed=0):
        raise NotImplementedError(
            "string_method is deferred to confirmatory use on flagged pairs "
            "only, after the Phase 1 routing matrix identifies candidates -- "
            "see this module's docstring and Strategic_Blind_Spots_Analysis.md #4."
        )
