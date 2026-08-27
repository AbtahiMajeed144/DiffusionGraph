"""
Post-hoc analysis of a gate sweep's resume cache -- works on a FULLY
COMPLETE run or a PARTIAL one (the 45x3x3 sweep may still be running).
Reads directly from results/gate/<run_name>/cache/, no recomputation of
any path, no need to wait for the sweep to finish.

Motivation (see conversation / SEED_semantic_class_graph.md §3.3, §6):
C(A,B) > tau=0.5 is a strong, principled bar -- since all 10 class
probabilities sum to 1, only one class can ever exceed 0.5, so crossing it
means the third class became the model's actual #1 prediction, beating
BOTH endpoints AND everything else outright. But it says nothing about the
more relative case: a third class clearly dominating its non-endpoint
peers, or even becoming the true argmax over all 10 classes, while sitting
well under 0.5 (e.g. A=0.3, B=0.2, C=0.36 -- C is already the argmax, but
C(A,B)'s hard threshold would silently miss this). This script computes,
directly from the cached raw softmax trajectories:
  - the existing strict metric (C(A,B) > tau, already stored)
  - is_full_argmax: does the peak "other" class actually win the FULL
    10-way vote at that point, not just the other-8-class subset?
    (weaker than >0.5 -- can be true at much lower absolute values)
  - margin_runnerup: peak class's lead over the 2nd-best non-endpoint class
  - margin_over_best_endpoint: peak class's lead over max(p(A), p(B))

Also computes, per combo (same "free" cost -- everything comes from the
already-cached softmax, no new experiments):
  - p_ab_at_peak: max(p(A), p(B)) at the SAME point where the "other"
    class peaked. High = the image was still confidently A-or-B even at
    its most routed-looking moment (realistic, just not evidence of
    routing). Low = neither endpoint was confident there either.
  - min_max_ab_over_path: the WORST point across the entire path for
    endpoint confidence -- min over t of max(p(A), p(B)). If this stays
    reasonably high, the path never left "recognizably A or B" territory.
    If it's low (near the ~0.1 uniform-over-10-classes floor), that's the
    signature of a genuinely ambiguous/OOD stretch -- nothing (not even
    the endpoints) gets confident recognition there.
These directly test whether "no clear third class" means "still looks
like A or B" (fine for realism, just not routing) vs "genuinely
unrecognizable as anything" (a real realism problem for this path type).

--evaluators lets you restrict which evaluators feed the strict/
consistent-flip/consensus computation, e.g. to check whether
disagreement is partly an artifact of one weaker evaluator (a
from-scratch ViT trained to only ~70% clean-test accuracy will be a much
noisier vote than a 93%+ ResNet or CLIP, for reasons unrelated to
whether the image itself is realistic):
    --evaluators resnet50,clip_zeroshot

Usage:
    python scripts/analyze_cache.py --run-name phase1_gate_full
    python scripts/analyze_cache.py --run-name phase1_gate_full --tag real --path tangential_geodesic --sigma 2.0
    python scripts/analyze_cache.py --run-name phase1_gate_full --evaluators resnet50,clip_zeroshot
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from diffusiongraph.config import RESULTS_DIR, CIFAR10_CLASSES
from run_gate import get_class_pairs, _all_combo_keys  # reuse the exact enumeration run_gate.py itself uses

COMBO_RE = re.compile(
    r"^(?P<tag>real|permuted)__(?P<path>[a-z_]+)__sigma(?P<sigma>[^_]+)__pair(?P<a>\d+)-(?P<b>\d+)__seed(?P<seed>\d+)$"
)


def parse_combo_key(key: str):
    m = COMBO_RE.match(key)
    if not m:
        return None
    d = m.groupdict()
    sigma_key = "final" if d["sigma"] == "final" else d["sigma"].replace("p", ".")
    return {"tag": d["tag"], "path": d["path"], "sigma_key": sigma_key, "a": int(d["a"]), "b": int(d["b"]), "seed": int(d["seed"])}


def analyze_combo(softmax_by_evaluator: dict, class_a: int, class_b: int) -> dict:
    """Mirrors eval/routing.py's compute_C to find the same (t*, c*) peak,
    then adds the relative-dominance metrics described in the module
    docstring. softmax_by_evaluator: {name: np.ndarray [T,B,10]}."""
    results = {}
    for name, arr in softmax_by_evaluator.items():
        mean_sm = arr.mean(axis=1)  # [T,10]
        num_classes = mean_sm.shape[1]
        mask = np.ones(num_classes, dtype=bool)
        mask[class_a] = False
        mask[class_b] = False
        other = mean_sm[:, mask]  # [T, 8]
        other_ids = np.arange(num_classes)[mask]

        t_idx, c_idx = np.unravel_index(other.argmax(), other.shape)
        peak_value = float(other[t_idx, c_idx])
        peak_class = int(other_ids[c_idx])

        full_row = mean_sm[t_idx]
        is_full_argmax = int(full_row.argmax()) == peak_class

        other_sorted = np.sort(other[t_idx])[::-1]
        runnerup = float(other_sorted[1]) if len(other_sorted) > 1 else 0.0
        best_endpoint = float(max(full_row[class_a], full_row[class_b]))

        # max(p(A), p(B)) at EVERY t, not just the "other"-peak t -- lets
        # us ask: across the whole path, does endpoint confidence ever
        # fully collapse (genuinely OOD stretch), or does the path stay
        # "recognizably A or B" throughout even where routing evidence is
        # weakest?
        max_ab_per_t = np.maximum(mean_sm[:, class_a], mean_sm[:, class_b])  # [T]

        results[name] = dict(
            peak_value=peak_value,
            peak_class=peak_class,
            is_full_argmax=bool(is_full_argmax),
            margin_runnerup=peak_value - runnerup,
            margin_over_best_endpoint=peak_value - best_endpoint,
            p_ab_at_peak=best_endpoint,                        # endpoint confidence AT the other-class peak moment
            min_max_ab_over_path=float(max_ab_per_t.min()),    # worst point anywhere on the path for endpoint confidence
            avg_max_ab_over_path=float(max_ab_per_t.mean()),
        )
    return results


def load_run_config(cache_dir: Path) -> SimpleNamespace:
    path = cache_dir / "run_config.json"
    if not path.exists():
        raise FileNotFoundError(f"No run_config.json in {cache_dir} -- has the sweep produced any output yet?")
    d = json.loads(path.read_text())
    return SimpleNamespace(
        class_pair_mode=d["class_pair_mode"],
        poc_pairs=tuple(tuple(p) for p in d["poc_pairs"]),
        routing_sigmas=tuple(d["routing_sigmas"]),
        enabled_paths=tuple(d["enabled_paths"]),
        routing_seeds=tuple(d["routing_seeds"]),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--tag", default="real", choices=["real", "permuted"])
    p.add_argument("--path", default="tangential_geodesic")
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument(
        "--evaluators", default=None,
        help="Comma-separated subset of evaluator names to use for strict/consistent-flip/"
             "consensus/p(A,B) computation, e.g. resnet50,clip_zeroshot -- drops the rest "
             "(e.g. a weaker evaluator whose disagreement may reflect its own accuracy, not "
             "the image's realism). Default: use all evaluators present in the cache."
    )
    args = p.parse_args()
    eval_filter = set(n.strip() for n in args.evaluators.split(",")) if args.evaluators else None

    out_dir = RESULTS_DIR / "gate" / args.run_name
    cache_dir = out_dir / "cache"
    cfg = load_run_config(cache_dir)
    expected_n_seeds = len(cfg.routing_seeds)

    class_pairs = get_class_pairs(cfg)
    expected_for_path = {k for k in _all_combo_keys(cfg, class_pairs, args.tag) if f"__{args.path}__" in k}
    present_for_path = sorted(
        k for k in expected_for_path
        if (cache_dir / f"{k}.json").exists() and (cache_dir / f"{k}.npz").exists()
    )

    n_expected, n_present = len(expected_for_path), len(present_for_path)
    status = "COMPLETE" if n_present == n_expected else f"PARTIAL -- {n_present}/{n_expected} ({100*n_present/max(n_expected,1):.1f}%)"
    print(f"=== Post-hoc analysis: run_name={args.run_name} tag={args.tag} path={args.path} ===")
    print(f"Coverage: {status}")
    if eval_filter:
        print(f"Evaluators: {sorted(eval_filter)} (filtered from cache)")
    print()

    grouped = defaultdict(list)
    for key in present_for_path:
        info = parse_combo_key(key)
        payload = json.loads((cache_dir / f"{key}.json").read_text())
        npz = np.load(cache_dir / f"{key}.npz")
        evaluator_names = [n for n in npz.files if eval_filter is None or n in eval_filter]
        evaluator_c = {n: v for n, v in payload["evaluator_c"].items() if n in evaluator_names}
        extra = analyze_combo({name: npz[name] for name in evaluator_names}, info["a"], info["b"])
        grouped[(info["sigma_key"], info["a"], info["b"])].append(
            {"seed": info["seed"], "evaluator_c": evaluator_c, "extra": extra}
        )

    if not grouped:
        print("No combos found yet for this tag/path -- nothing to analyze.")
        return

    rows = []
    for (sigma_key, a, b), seed_results in sorted(grouped.items(), key=lambda x: (str(x[0][0]), x[0][1], x[0][2])):
        n_seeds = len(seed_results)
        strict = all(v >= args.tau for r in seed_results for v in r["evaluator_c"].values())
        all_flip = all(r["extra"][ev]["is_full_argmax"] for r in seed_results for ev in r["extra"])
        peak_classes = {r["extra"][ev]["peak_class"] for r in seed_results for ev in r["extra"]}
        consistent_flip = all_flip and len(peak_classes) == 1
        all_extra = [r["extra"][ev] for r in seed_results for ev in r["extra"]]
        avg_margin = float(np.mean([e["margin_runnerup"] for e in all_extra]))
        avg_p_ab_at_peak = float(np.mean([e["p_ab_at_peak"] for e in all_extra]))
        avg_min_max_ab = float(np.mean([e["min_max_ab_over_path"] for e in all_extra]))
        consensus = CIFAR10_CLASSES[peak_classes.pop()] if len(peak_classes) == 1 else "disagree:" + ",".join(sorted(CIFAR10_CLASSES[c] for c in peak_classes))
        partial = n_seeds < expected_n_seeds
        rows.append(dict(
            pair=f"{CIFAR10_CLASSES[a]}-{CIFAR10_CLASSES[b]}", sigma=sigma_key,
            n_seeds=n_seeds, expected_n_seeds=expected_n_seeds, partial=partial,
            strict_routing=strict, consistent_argmax_flip=consistent_flip,
            avg_margin_runnerup=round(avg_margin, 4),
            avg_p_ab_at_peak=round(avg_p_ab_at_peak, 4),
            avg_min_max_ab_over_path=round(avg_min_max_ab, 4),
            consensus_class=consensus,
        ))

    hdr = (f"{'pair':<20} {'sigma':<7} {'seeds':<9} {'strict':<8} {'flip':<7} {'margin':<8} "
           f"{'p(AB)@peak':<11} {'min_p(AB)':<10} consensus_class")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        seeds_str = f"{r['n_seeds']}/{r['expected_n_seeds']}" + (" *" if r["partial"] else "")
        print(f"{r['pair']:<20} {str(r['sigma']):<7} {seeds_str:<9} {str(r['strict_routing']):<8} "
              f"{str(r['consistent_argmax_flip']):<7} {r['avg_margin_runnerup']:<8} "
              f"{r['avg_p_ab_at_peak']:<11} {r['avg_min_max_ab_over_path']:<10} {r['consensus_class']}")
    if any(r["partial"] for r in rows):
        print("\n* = fewer seeds than the profile's routing_seeds count -- preliminary, not a confirmed result yet.")
    print(
        "\np(AB)@peak = avg max(p(A),p(B)) at the SAME moment the 'other' class peaked -- high means the "
        "image still looked like an endpoint even there (realistic, just not routed).\n"
        "min_p(AB)  = avg (across evaluators/seeds) of the WORST point anywhere on the path for endpoint "
        "confidence -- low means there's a stretch where neither A nor B (nor anything else, per the disagreement "
        "pattern) gets confidently recognized -- the genuinely-OOD signature."
    )

    complete_rows = [r for r in rows if not r["partial"]]
    n_strict = sum(r["strict_routing"] for r in complete_rows)
    n_flip = sum(r["consistent_argmax_flip"] for r in complete_rows)
    avg_p_ab_at_peak_all = float(np.mean([r["avg_p_ab_at_peak"] for r in complete_rows])) if complete_rows else float("nan")
    avg_min_p_ab_all = float(np.mean([r["avg_min_max_ab_over_path"] for r in complete_rows])) if complete_rows else float("nan")
    print(f"\nAmong {len(complete_rows)} pairs with full seed coverage for {args.path!r}:")
    print(f"  strict routing events (C(A,B)>tau on all evaluators/seeds): {n_strict}")
    print(f"  consistent argmax-flip events (weaker, relative-dominance): {n_flip}")
    print(f"  avg p(A or B) at the other-class peak moment: {avg_p_ab_at_peak_all:.4f}")
    print(f"  avg worst-point-on-path p(A or B):            {avg_min_p_ab_all:.4f}")

    summary_path = out_dir / f"posthoc_{args.tag}_{args.path}.json"
    summary_path.write_text(json.dumps({"coverage": status, "rows": rows}, indent=2))
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
