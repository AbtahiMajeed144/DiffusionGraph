"""
GRAPH_BARRIER_EXPERIMENT.md Stages 1-3: node set, pixel-kNN graph with
realism-min edge weights, and union-find barrier extraction.

Realism R is the far-validated feature-kNN score (Stage-0 far-AUROC 0.92), NOT
the fine near-OOD score that failed the Stage-0 gate -- we only ask R the coarse
question it can answer (on-manifold vs off), never the fine one it can't.

Design-grounded change: cross/same-class slerp interpolant nodes are seeded across
a RANGE of sigma (not just 0.5), because the manifold measurement showed sigma=0.5
midpoints are off-manifold while higher-sigma ones are realistic-but-drifting. The
graph must be given the realistic candidates or a low tau* is just a coverage
artifact (design 1.2 / 6.1).

Node types (design 1.1), all stored as clean decoded [-1,1]^{3x32x32}:
  anchors           real test images, define class membership + path endpoints
  filler            unconditional EDM samples (alternative routes / hubs)
  cross-interp      slerp-in-noise (per sigma, decoded) + pixel-linear, A!=B
  same-interp       same, A==A' (within-class null, design 5.1 / 6.1)
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
from typing import Callable, Optional

import numpy as np
import torch

from diffusiongraph.data.cifar10 import CIFAR10Canonical
from diffusiongraph.paths.slerp_noise import slerp
from diffusiongraph.paths.base import forward_diffuse


@dataclass
class NodeSet:
    images: torch.Tensor            # [N,3,32,32] cpu, clean [-1,1]
    anchor_class: np.ndarray        # [N] int, class 0..9 for anchors else -1
    provenance: list                # [N] dicts (type, a, b, t, path, sigma)
    R: Optional[np.ndarray] = None  # [N] realism, filled by score_nodes


# --------------------------------------------------------------------------
# Stage 1: node set
# --------------------------------------------------------------------------
@torch.no_grad()
def build_node_set(denoiser, *, anchors_per_class=32, n_filler=2000,
                   pairs_per_classpair=8, n_t=17, interp_sigmas=(0.5, 2.0, 8.0),
                   include_pixel_linear=True, decode_steps=18, seed=0, device="cuda",
                   n_class_pairs=None):
    g = torch.Generator().manual_seed(seed)
    test = CIFAR10Canonical(train=False, download=True)
    imgs, anchor_class, prov = [], [], []

    def add(x, cls, meta):
        imgs.append(x.cpu()); anchor_class.append(cls); prov.append(meta)

    # anchors
    anchor_pool = {}
    for c in range(10):
        idx = test.indices_for_class(c)[:anchors_per_class]
        anchor_pool[c] = [test[i][0] for i in idx]
        for j, im in enumerate(anchor_pool[c]):
            add(im, c, {"type": "anchor", "a": c, "b": c, "t": 0.0, "path": "real", "sigma": None})

    # filler (unconditional samples)
    made = 0
    while made < n_filler:
        b = min(200, n_filler - made)
        s = denoiser.sample(class_idx=None, batch_size=b, num_steps=decode_steps, seed=seed + made)
        for k in range(b):
            add(s[k], -1, {"type": "filler", "a": -1, "b": -1, "t": None, "path": "uncond", "sigma": None})
        made += b

    t_values = torch.linspace(0, 1, n_t)[1:-1]  # interior only (endpoints are anchors)

    def _pick(c):
        return anchor_pool[c][int(torch.randint(len(anchor_pool[c]), (1,), generator=g))]

    def interpolants(ca, cb, tag):
        for _ in range(pairs_per_classpair):
            xa, xb = _pick(ca), _pick(cb)
            xa_d = xa.to(device); xb_d = xb.to(device)
            for sig in interp_sigmas:
                a_s = forward_diffuse(xa_d[None], sig, generator=g)
                b_s = forward_diffuse(xb_d[None], sig, generator=g)
                # batch ALL t-values of this (endpoint-pair, sigma) into one decode
                mids = torch.cat([slerp(a_s, b_s, float(t)) for t in t_values], 0)
                dec = denoiser.denoise_to_clean(mids, sig, class_labels=None, num_steps=decode_steps)
                for j, t in enumerate(t_values):
                    add(dec[j], -1, {"type": tag, "a": ca, "b": cb, "t": float(t), "path": "slerp", "sigma": float(sig)})
            if include_pixel_linear:
                for t in t_values:
                    add(((1 - float(t)) * xa + float(t) * xb).clamp(-1, 1), -1,
                        {"type": tag, "a": ca, "b": cb, "t": float(t), "path": "pixel", "sigma": None})

    pairs = [(a, b) for a in range(10) for b in range(a + 1, 10)]
    if n_class_pairs is not None:
        pairs = pairs[:n_class_pairs]
    for (ca, cb) in pairs:
        interpolants(ca, cb, "cross")
    for c in range(10):
        interpolants(c, c, "same")

    return NodeSet(images=torch.stack(imgs, 0),
                   anchor_class=np.asarray(anchor_class, dtype=np.int64),
                   provenance=prov)


def score_nodes(ns: NodeSet, realism_fn: Callable[[torch.Tensor], np.ndarray]) -> NodeSet:
    ns.R = np.asarray(realism_fn(ns.images))
    return ns


# --------------------------------------------------------------------------
# Stage 2: pixel-L2 kNN graph (+ MST union for connectivity)
# --------------------------------------------------------------------------
def _pairwise_knn(flat: torch.Tensor, k: int, device: str, chunk=1024):
    N = flat.shape[0]
    flat = flat.to(device)
    nbrs = np.zeros((N, k), dtype=np.int64)
    dists = np.zeros((N, k), dtype=np.float32)
    for i in range(0, N, chunk):
        d = torch.cdist(flat[i:i + chunk], flat)          # [c, N]
        d[torch.arange(d.shape[0]), torch.arange(i, i + d.shape[0])] = float("inf")
        dk, ik = torch.topk(d, k, dim=1, largest=False)
        nbrs[i:i + d.shape[0]] = ik.cpu().numpy()
        dists[i:i + d.shape[0]] = dk.cpu().numpy()
    return nbrs, dists


def build_graph(ns: NodeSet, k=20, device="cuda"):
    """Returns edge dict {(u,v): dist} symmetric, with an MST union guaranteeing
    one connected component."""
    flat = ns.images.flatten(1)
    nbrs, dists = _pairwise_knn(flat, k, device)
    edges = {}
    for u in range(nbrs.shape[0]):
        for j in range(nbrs.shape[1]):
            v = int(nbrs[u, j]); d = float(dists[u, j])
            key = (u, v) if u < v else (v, u)
            if key not in edges or d < edges[key]:
                edges[key] = d
    # MST union for connectivity: add cheapest cross-component edges from the kNN
    # candidate pool; if still disconnected, connect components by nearest centroids.
    edges = _ensure_connected(ns, edges, flat, device)
    return edges


def _ensure_connected(ns, edges, flat, device):
    N = ns.images.shape[0]
    parent = list(range(N))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb; return True
        return False
    for (u, v) in edges:
        union(u, v)
    roots = sorted({find(i) for i in range(N)})
    if len(roots) == 1:
        return edges
    # connect components greedily by nearest representative (root) pixel distance
    reps = torch.stack([flat[r] for r in roots]).to(device)
    d = torch.cdist(reps, reps); d.fill_diagonal_(float("inf"))
    order = torch.argsort(d.flatten())
    for idx in order.tolist():
        i, j = idx // len(roots), idx % len(roots)
        if union(roots[i], roots[j]):
            key = (roots[i], roots[j]) if roots[i] < roots[j] else (roots[j], roots[i])
            edges[key] = float(d[i, j].item())
        if len({find(r) for r in roots}) == 1:
            break
    return edges


# --------------------------------------------------------------------------
# Stage 2.2/2.3: realism-min edge weights, segment-sampled, lazy-refined
# --------------------------------------------------------------------------
def _segment_min_R(ns, realism_fn, u, v, delta, cache):
    key = (u, v) if u < v else (v, u)
    if key in cache:
        return cache[key]
    iu, iv = ns.images[u], ns.images[v]
    dist = float(torch.linalg.norm((iu - iv).flatten()))
    S = max(2, math.ceil(dist / delta))
    ss = torch.linspace(0, 1, S)
    pts = torch.stack([(1 - float(s)) * iu + float(s) * iv for s in ss], 0)
    w = float(np.min(realism_fn(pts)))
    cache[key] = w
    return w


def barrier_matrix(ns: NodeSet, edges: dict, realism_fn, *, delta, refine="lazy",
                   max_rounds=6, tol=1e-4, verbose=True):
    """Iterate: optimistic min(R) weights -> union-find barriers -> refine the 45
    bottleneck PATHS with segment sampling -> repeat until paths stabilize.
    Returns (tau [10,10], history)."""
    R = ns.R
    seg_cache = {}
    # optimistic weights (free)
    w = {e: min(R[e[0]], R[e[1]]) for e in edges}
    if refine == "full":
        for e in edges:
            w[e] = _segment_min_R(ns, realism_fn, e[0], e[1], delta, seg_cache)
        tau, _ = _union_find_barriers(ns, w)
        return tau, [tau.copy()], w

    history = []
    prev_tau = None
    for rnd in range(max_rounds):
        tau, paths = _union_find_barriers(ns, w, return_paths=True)
        history.append(tau.copy())
        # refine every edge on all 45 bottleneck paths
        to_refine = set()
        for pth in paths.values():
            for a, b in zip(pth[:-1], pth[1:]):
                key = (a, b) if a < b else (b, a)
                if key in w:
                    to_refine.add(key)
        changed = False
        for e in to_refine:
            nw = _segment_min_R(ns, realism_fn, e[0], e[1], delta, seg_cache)
            if nw < w[e] - 1e-9:
                w[e] = nw; changed = True
        # converge on the tau* VALUES, not the path identities: in a dense graph the
        # max-min bottleneck route keeps switching among near-tied alternatives even
        # after tau* has settled, so a path-identity criterion never trips.
        tau_next = _union_find_barriers(ns, w)[0]
        iu = np.triu_indices(10, 1)
        d = tau_next[iu] - tau[iu]
        max_delta = float(np.nanmax(np.abs(d))) if np.isfinite(d).any() else 0.0
        if verbose:
            print(f"  refine round {rnd}: {len(to_refine)} path edges, changed={changed}, "
                  f"max|Δτ*|={max_delta:.5f}")
        if not changed or max_delta < tol:
            break
        prev_tau = tau_next
    tau, _ = _union_find_barriers(ns, w)
    history.append(tau.copy())
    return tau, history, w


# --------------------------------------------------------------------------
# Stage 3: union-find single-linkage barrier extraction
# --------------------------------------------------------------------------
def _union_find_barriers(ns: NodeSet, w: dict, return_paths=False):
    """tau*(A,B) = weight of the edge that first connects an A-anchor component to
    a B-anchor component, scanning edges by descending weight (max-min path)."""
    N = ns.images.shape[0]
    parent = list(range(N))
    cls_mask = [0] * N
    for i in range(N):
        c = ns.anchor_class[i]
        if c >= 0:
            cls_mask[i] = 1 << int(c)
    # for path reconstruction: union tree edges
    link = {}  # child_root -> (parent_root, edge)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    tau = np.full((10, 10), np.nan)
    joined = np.zeros((10, 10), dtype=bool)
    edge_of = {}  # (A,B) -> connecting edge

    for (u, v), weight in sorted(w.items(), key=lambda kv: -kv[1]):
        ru, rv = find(u), find(v)
        if ru == rv:
            continue
        mu, mv = cls_mask[ru], cls_mask[rv]
        # record new cross-class connections BEFORE merging
        new_pairs = []
        if mu and mv:
            for A in range(10):
                if not (mu >> A) & 1:
                    continue
                for B in range(10):
                    if A < B and ((mv >> B) & 1) and not joined[A, B]:
                        new_pairs.append((A, B))
                    if B < A and ((mv >> B) & 1) and not joined[B, A]:
                        new_pairs.append((B, A))
                # also mv has A side / mu has B side
            for A in range(10):
                if not (mv >> A) & 1:
                    continue
                for B in range(10):
                    if A < B and ((mu >> B) & 1) and not joined[A, B]:
                        new_pairs.append((A, B))
                    if B < A and ((mu >> B) & 1) and not joined[B, A]:
                        new_pairs.append((B, A))
        # union
        parent[ru] = rv
        cls_mask[rv] = mu | mv
        link[ru] = (rv, (u, v))
        for (A, B) in set(new_pairs):
            if not joined[A, B]:
                tau[A, B] = tau[B, A] = weight
                joined[A, B] = joined[B, A] = True
                edge_of[(A, B)] = (u, v)

    paths = {}
    if return_paths:
        # bottleneck path per pair = anchors' path in the max-weight spanning forest.
        paths = _bottleneck_paths(ns, w)
    return (tau, paths) if return_paths else (tau, {})


def _bottleneck_paths(ns: NodeSet, w: dict):
    """Max-spanning-tree of the graph; for each class pair return the tree path
    between the first A-anchor and first B-anchor (edges to refine)."""
    N = ns.images.shape[0]
    parent = list(range(N))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    adj = {i: [] for i in range(N)}
    for (u, v), weight in sorted(w.items(), key=lambda kv: -kv[1]):
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
            adj[u].append(v); adj[v].append(u)
    anchors = {}
    for c in range(10):
        idx = np.where(ns.anchor_class == c)[0]
        if len(idx):
            anchors[c] = int(idx[0])
    paths = {}
    for A in range(10):
        for B in range(A + 1, 10):
            if A in anchors and B in anchors:
                p = _tree_path(adj, anchors[A], anchors[B])
                if p:
                    paths[(A, B)] = p
    return paths


def tau_from_node_R(ns: NodeSet, edges: dict, Rvec: np.ndarray, node_subset: set = None):
    """tau* using optimistic node-based weights w = min(R[u],R[v]) (no segment
    refinement). If node_subset is given, only edges with BOTH endpoints in it are
    used. Cheap -- for the shuffled-R null (5.2) and filler-removal diagnostics."""
    if node_subset is None:
        w = {e: min(Rvec[e[0]], Rvec[e[1]]) for e in edges}
    else:
        w = {e: min(Rvec[e[0]], Rvec[e[1]]) for e in edges
             if e[0] in node_subset and e[1] in node_subset}
    tau, _ = _union_find_barriers(ns, w)
    return tau


def shuffled_R_null(ns: NodeSet, edges: dict, n_perm=50, seed=0):
    """Design 5.2: permute R across nodes, recompute tau* (node-weighted), and ask
    whether the REAL tau* spread exceeds the topology-only null. Returns dict with
    the real cross-pair spread (IQR), the null spread distribution, a percentile,
    and the median |Spearman| of real-vs-shuffled (should be ~0 if structure is
    real)."""
    from scipy.stats import spearmanr
    iu = np.triu_indices(10, 1)
    real_tau = tau_from_node_R(ns, edges, ns.R)
    real_cross = real_tau[iu]
    def spread(v):
        v = v[~np.isnan(v)]
        return float(np.subtract(*np.percentile(v, [75, 25]))) if len(v) > 2 else np.nan
    real_spread = spread(real_cross)
    rng = np.random.default_rng(seed)
    null_spreads, rhos = [], []
    for _ in range(n_perm):
        Rp = ns.R.copy(); rng.shuffle(Rp)
        t = tau_from_node_R(ns, edges, Rp)[iu]
        null_spreads.append(spread(t))
        m = ~np.isnan(real_cross) & ~np.isnan(t)
        if m.sum() > 2:
            rhos.append(abs(spearmanr(real_cross[m], t[m]).correlation))
    null_spreads = np.array(null_spreads)
    pct = float((null_spreads < real_spread).mean() * 100)
    return {"real_spread_iqr": real_spread, "null_spread_median": float(np.nanmedian(null_spreads)),
            "null_spread_p95": float(np.nanpercentile(null_spreads, 95)),
            "real_spread_percentile_vs_null": pct,
            "median_abs_spearman_real_vs_shuffled": float(np.median(rhos)) if rhos else np.nan}


def filler_removed_tau(ns: NodeSet, edges: dict):
    """Filler-hub diagnosis: recompute tau* (node-weighted) with all filler nodes
    removed. If cross-class tau* collapses, the 'connectivity' was filler-hopping."""
    keep = {i for i in range(ns.images.shape[0]) if ns.provenance[i]["type"] != "filler"}
    iu = np.triu_indices(10, 1)
    full = tau_from_node_R(ns, edges, ns.R)[iu]
    nofill = tau_from_node_R(ns, edges, ns.R, node_subset=keep)[iu]
    return {"cross_median_full": float(np.nanmedian(full)),
            "cross_median_no_filler": float(np.nanmedian(nofill)),
            "delta": float(np.nanmedian(nofill) - np.nanmedian(full)),
            "n_pairs_disconnected_no_filler": int(np.isnan(nofill).sum() - np.isnan(full).sum())}


def maxmin_over_subgraph(w: dict, allowed: set, src: set, dst: set) -> float:
    """Max-min (bottleneck) realism between node-sets src and dst using only edges
    whose BOTH endpoints are in `allowed`. Returns the weight of the edge that
    first connects any src node to any dst node, scanning by descending weight."""
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def tag(x):  # 1 if src side, 2 if dst side, 0 otherwise; carried on the root
        return (1 if x in src else 0) | (2 if x in dst else 0)
    side = {}
    for n in allowed:
        r = find(n); side[r] = side.get(r, 0) | tag(n)
    for (u, v), weight in sorted(w.items(), key=lambda kv: -kv[1]):
        if u not in allowed or v not in allowed:
            continue
        ru, rv = find(u), find(v)
        if ru == rv:
            continue
        merged = side.get(ru, 0) | side.get(rv, 0)
        parent[ru] = rv; side[rv] = merged
        if merged == 3:
            return float(weight)
    return float("nan")


def within_class_floor(ns: NodeSet, w: dict, fair: bool = False) -> np.ndarray:
    """Design 5.1: split each class's anchors in two halves and compute the
    within-class barrier. Returns [10].

    fair=False (design 5.1 literal): between-region restricted to that class's own
    same-class interpolant nodes -- tests whether same-class interpolants alone stay
    realistic. Low => the interpolants are off-manifold (not a P6 result).
    fair=True: allow the FULL node set (filler + all interpolants) between the two
    halves, exactly as cross-class tau* does -- the apples-to-apples P6 comparison.
    within(fair) should EXCEED cross-class tau* if class basins are resolved."""
    all_nodes = set(range(ns.images.shape[0]))
    same_of = {c: set() for c in range(10)}
    for i, p in enumerate(ns.provenance):
        if p["type"] == "same" and p["a"] == p["b"]:
            same_of[p["a"]].add(i)
    floors = np.full(10, np.nan)
    for c in range(10):
        anc = np.where(ns.anchor_class == c)[0].tolist()
        if len(anc) < 2:
            continue
        h = len(anc) // 2
        src, dst = set(anc[:h]), set(anc[h:])
        # exclude the OTHER classes' anchors so cross-class hops can't shortcut
        other_anchors = {i for i in np.where(ns.anchor_class >= 0)[0] if i not in src and i not in dst}
        allowed = (all_nodes - other_anchors) if fair else (src | dst | same_of[c])
        floors[c] = maxmin_over_subgraph(w, allowed, src, dst)
    return floors


def comparison_graphs(ns: NodeSet, tau: np.ndarray, clip_extractor=None):
    """Design 7 / P2: build cheap alternative 10x10 class-affinity matrices and
    Spearman-correlate against tau* on the 45 off-diagonal entries. If tau* is a
    monotone re-derivation of pixel or CLIP centroid distance (rho > 0.9), the
    barrier machinery adds nothing.

    G_pixel   : -||pixel centroid_A - centroid_B|| over anchors
    G_clip    : -||CLIP centroid_A - centroid_B|| (if clip_extractor given)
    Returns {name: (matrix, spearman_vs_tau)}."""
    from scipy.stats import spearmanr
    iu = np.triu_indices(10, 1)
    anchors = {c: np.where(ns.anchor_class == c)[0] for c in range(10)}

    def affinity_from(feats_per_class):
        cen = np.stack([feats_per_class[c].mean(0) for c in range(10)])
        G = np.full((10, 10), np.nan)
        for a in range(10):
            for b in range(10):
                if a != b:
                    G[a, b] = -float(np.linalg.norm(cen[a] - cen[b]))
        return G

    out = {}
    pix = {c: ns.images[anchors[c]].flatten(1).numpy() for c in range(10)}
    out["G_pixel"] = affinity_from(pix)
    if clip_extractor is not None:
        import torch as _t
        clip_f = {}
        for c in range(10):
            with _t.no_grad():
                clip_f[c] = clip_extractor.features(ns.images[anchors[c]]).cpu().numpy()
        out["G_clip"] = affinity_from(clip_f)

    res = {}
    for name, G in out.items():
        m = ~np.isnan(tau[iu]) & ~np.isnan(G[iu])
        rho = spearmanr(tau[iu][m], G[iu][m]).correlation if m.sum() > 2 else float("nan")
        res[name] = (G, float(rho))
    return res


def route_provenance(ns: NodeSet, w: dict) -> dict:
    """Design 4.1: for each cross pair, classify the bottleneck path's interior
    nodes -> fraction filler / own-pair interpolant / other-pair interpolant.
    Diagnoses the 'all routes hop through filler' failure mode."""
    paths = _bottleneck_paths(ns, w)
    comp = {}
    for (A, B), pth in paths.items():
        interior = pth[1:-1]
        n = max(1, len(interior))
        filler = own = other = anchor = 0
        for node in interior:
            p = ns.provenance[node]
            if p["type"] == "filler":
                filler += 1
            elif p["type"] == "anchor":
                anchor += 1
            elif {p["a"], p["b"]} == {A, B}:
                own += 1
            else:
                other += 1
        comp[(A, B)] = {"len": len(interior), "filler": filler / n, "own_pair": own / n,
                        "other_pair": other / n, "anchor": anchor / n}
    return comp


def _tree_path(adj, src, dst):
    from collections import deque
    prev = {src: None}
    q = deque([src])
    while q:
        x = q.popleft()
        if x == dst:
            break
        for y in adj[x]:
            if y not in prev:
                prev[y] = x; q.append(y)
    if dst not in prev:
        return None
    path = []
    x = dst
    while x is not None:
        path.append(x); x = prev[x]
    return path[::-1]
