"""
Phase 0 Audit -- the 5 checks from the suggested audit report (see
PHASE0_AUDIT.md). All read-only against the existing cache, no GPU needed.

Checks implemented here (2, 3, 4, and an empirical check for 5):
  2. Histogram of C(A,B) across all pairs x seeds x evaluators -- is it a
     smooth unimodal mass near 0.45-0.50 (== "no separation from null",
     the honest characterization) rather than "0 events" (which implies
     a bimodal/separated distribution that just didn't cross a line)?
  3. Marginal distribution of the "other-class" argmax over ALL pairs --
     if one class (e.g. airplane) dominates GLOBALLY, any single pair's
     consensus on that class is a classifier prior, not evidence of
     routing specific to that pair.
  4. vit_base ablation on the CONTINUOUS C(A,B) distribution per
     evaluator, not on event counts (comparing 0 to 0 established
     nothing -- SEED's own routing_threshold_tau=0.5 bar was never
     crossed by ANY evaluator in the sweep so far, so an event-count
     ablation is uninformative by construction).
  5. Empirical check for the real/permuted non-independence argument
     (see PHASE0_AUDIT.md's code-trace finding): for paths 2-4
     (unconditional -- slerp_noise, tangential_geodesic), do real-pair
     (a,b)'s cached C values correlate with permuted-pair (pi(a),pi(b))'s
     -- i.e. the SAME underlying real class pair, reached via the
     permutation -- more than they correlate with an unrelated pair? If
     so, that's direct empirical confirmation the two sweeps are testing
     the same underlying experiments, not independent ones.
     (Check 1 -- which tensor reaches the evaluators -- is a pure code
     read, not data-dependent; done directly against source, not here.)

Usage:
    python scripts/audit_cache.py --run-name phase1_gate_full
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from diffusiongraph.config import RESULTS_DIR, CIFAR10_CLASSES
from run_gate import get_class_pairs, _all_combo_keys
from analyze_cache import parse_combo_key, load_run_config


def collect_all(cache_dir: Path, cfg, tag: str):
    """Returns a list of dicts, one per cached combo for this tag, with
    parsed key info + the raw evaluator_c / evaluator_argmax_class."""
    class_pairs = get_class_pairs(cfg)
    keys = _all_combo_keys(cfg, class_pairs, tag)
    records = []
    for key in keys:
        json_path = cache_dir / f"{key}.json"
        if not json_path.exists():
            continue
        info = parse_combo_key(key)
        payload = json.loads(json_path.read_text())
        records.append({**info, "evaluator_c": payload["evaluator_c"], "evaluator_argmax_class": payload["evaluator_argmax_class"]})
    return records


def check2_histogram(records, tau=0.5):
    print("\n--- Check 2: histogram of C(A,B), all pairs x seeds x evaluators ---")
    vals = [v for r in records for v in r["evaluator_c"].values()]
    if not vals:
        print("  (no data)")
        return
    vals = np.array(vals)
    bins = np.arange(0.0, 1.05, 0.05)
    hist, edges = np.histogram(vals, bins=bins)
    n = len(vals)
    for h, lo in zip(hist, edges[:-1]):
        bar = "#" * int(60 * h / max(hist.max(), 1))
        marker = " <== tau" if lo <= tau < lo + 0.05 else ""
        print(f"  [{lo:.2f},{lo+0.05:.2f}) {h:5d} ({100*h/n:5.1f}%) {bar}{marker}")
    print(f"  n={n}, mean={vals.mean():.4f}, median={np.median(vals):.4f}, std={vals.std():.4f}")
    frac_near_tau = ((vals > 0.40) & (vals < 0.50)).mean()
    frac_over_tau = (vals >= tau).mean()
    print(f"  fraction in (0.40, 0.50) [approaching but not crossing tau]: {100*frac_near_tau:.1f}%")
    print(f"  fraction >= tau={tau}: {100*frac_over_tau:.1f}%")
    if frac_over_tau < 0.02 and frac_near_tau > 0.15:
        print("  ==> HONEST READ: mass concentrated just below tau, unimodal -- this is 'no separation from")
        print("      null' (the whole distribution sits near the threshold), NOT '0 confirmed events' (which")
        print("      would imply a bimodal separation that merely failed to cross a line).")


def check3_marginal_argmax(records):
    print("\n--- Check 3: marginal distribution of the 'other-class' argmax over ALL pairs ---")
    counts = Counter()
    for r in records:
        for c in r["evaluator_argmax_class"].values():
            counts[c] += 1
    total = sum(counts.values())
    if total == 0:
        print("  (no data)")
        return
    for cls_idx, n in counts.most_common():
        pct = 100 * n / total
        flag = "  <== elevated, check per-pair claims involving this class carefully" if pct > 15 else ""
        print(f"  {CIFAR10_CLASSES[cls_idx]:<12} {n:5d} ({pct:5.1f}%){flag}")
    expected_uniform = 100 / 10
    print(f"  (uniform-over-10-classes baseline would be ~{expected_uniform:.1f}% each)")


def check4_evaluator_ablation(records):
    print("\n--- Check 4: per-evaluator CONTINUOUS C(A,B) distribution (not event counts) ---")
    by_eval = defaultdict(list)
    for r in records:
        for name, v in r["evaluator_c"].items():
            by_eval[name].append(v)
    for name, vals in sorted(by_eval.items()):
        vals = np.array(vals)
        print(f"  {name:<15} n={len(vals):4d}  mean={vals.mean():.4f}  median={np.median(vals):.4f}  "
              f"std={vals.std():.4f}  p10={np.percentile(vals,10):.4f}  p90={np.percentile(vals,90):.4f}")
    if "vit_base" in by_eval and len(by_eval) > 1:
        others = np.concatenate([v for k, v in by_eval.items() if k != "vit_base"])
        vb = np.array(by_eval["vit_base"])
        print(f"  vit_base mean {vb.mean():.4f} vs other evaluators' pooled mean {others.mean():.4f} "
              f"(diff={vb.mean()-others.mean():+.4f})")


def check5_permutation_independence(records_real, records_perm, permutation):
    """For unconditional paths only (slerp_noise, tangential_geodesic):
    pair real(a,b)'s C values vs permuted(pi(a),pi(b))'s C values -- the
    SAME underlying real class pair, reached via the permutation. If these
    correlate strongly, that's direct empirical evidence the two sweeps
    are the same experiment under different bookkeeping, not independent."""
    print("\n--- Check 5 (empirical): does permuted(pi(a),pi(b)) track real(a,b) for the SAME underlying pair? ---")
    print("    (unconditional paths only -- slerp_noise, tangential_geodesic; linear_condition has a separate, confirmed masking bug, see PHASE0_AUDIT.md)")
    real_by_key = {}
    for r in records_real:
        if r["path"] == "linear_condition":
            continue
        k = (r["path"], r["sigma_key"], r["a"], r["b"], r["seed"])
        real_by_key[k] = np.mean(list(r["evaluator_c"].values()))

    pairs = []
    for r in records_perm:
        if r["path"] == "linear_condition":
            continue
        pa, pb = int(permutation[r["a"]]), int(permutation[r["b"]])
        # NB: permuted dataset's sample_pairs(a,b) draws from real classes
        # pi^-1(a), pi^-1(b) -- so the "corresponding real pair" is
        # (pi^-1(a), pi^-1(b)), i.e. we need the INVERSE permutation here,
        # not the forward one applied to (a,b).
        inv = np.argsort(permutation)
        ra, rb = int(inv[r["a"]]), int(inv[r["b"]])
        ra, rb = min(ra, rb), max(ra, rb)
        k = (r["path"], r["sigma_key"], ra, rb, r["seed"])
        if k in real_by_key:
            perm_val = np.mean(list(r["evaluator_c"].values()))
            pairs.append((real_by_key[k], perm_val))

    if len(pairs) < 3:
        print(f"  Only {len(pairs)} matched (path,sigma,pair,seed) combos present in both sweeps so far -- need more overlap to say anything.")
        return
    real_vals, perm_vals = zip(*pairs)
    corr = float(np.corrcoef(real_vals, perm_vals)[0, 1])
    print(f"  n={len(pairs)} matched combos. correlation(real C, permuted C for the SAME real pair) = {corr:.3f}")
    if corr > 0.3:
        print("  ==> Positive correlation: permuted results for the same underlying real pair track the real")
        print("      results, consistent with the code-trace finding that these are structurally the same")
        print("      experiment under shuffled bookkeeping, not an independent identifiability test.")
    else:
        print("  ==> Weak/no correlation observed so far (may just be sample-size/seed-noise -- interpret with caution).")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    args = p.parse_args()

    out_dir = RESULTS_DIR / "gate" / args.run_name
    cache_dir = out_dir / "cache"
    cfg = load_run_config(cache_dir)

    records_real = collect_all(cache_dir, cfg, "real")
    records_perm = collect_all(cache_dir, cfg, "permuted")
    print(f"=== Phase 0 Audit: run_name={args.run_name} ===")
    print(f"Loaded {len(records_real)} real combos, {len(records_perm)} permuted combos from cache.")

    check2_histogram(records_real)
    check3_marginal_argmax(records_real)
    check4_evaluator_ablation(records_real)

    permutation = np.random.RandomState(cfg_permutation_seed(cache_dir)).permutation(10)
    check5_permutation_independence(records_real, records_perm, permutation)


def cfg_permutation_seed(cache_dir: Path) -> int:
    d = json.loads((cache_dir / "run_config.json").read_text())
    return d.get("permutation_seed", 1234)


if __name__ == "__main__":
    main()
