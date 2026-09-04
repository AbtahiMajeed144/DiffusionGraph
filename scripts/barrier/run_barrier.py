"""
Run GRAPH_BARRIER_EXPERIMENT Stages 1-3: build the node set, score realism,
build the pixel-kNN graph with realism-min edge weights, extract the 10x10
barrier matrix tau*, and (design 5.1) compare cross-class vs within-class floors.

Realism = feature-kNN to a real bank (far-validated). See barrier_graph.py header
for why this coarse score is the right one and the fine near-OOD score is not.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from diffusiongraph.config import get_profile, RESULTS_DIR, CHECKPOINTS_DIR
from diffusiongraph.utils.edm_loader import EDMDenoiser
from diffusiongraph.data.cifar10 import CIFAR10Canonical
from diffusiongraph.barrier.groups import _balanced_real
from diffusiongraph.barrier import scores as S
from diffusiongraph.barrier import barrier_graph as BG

CLASSES = ["plane", "auto", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="local_poc")
    ap.add_argument("--anchors-per-class", type=int, default=32)
    ap.add_argument("--n-filler", type=int, default=2000)
    ap.add_argument("--pairs-per-classpair", type=int, default=8)
    ap.add_argument("--n-t", type=int, default=17)
    ap.add_argument("--interp-sigmas", default="0.5,2.0,8.0")
    ap.add_argument("--no-pixel-linear", action="store_true")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--knn-k", type=int, default=5, help="k for the feature-kNN realism score")
    ap.add_argument("--decode-steps", type=int, default=18)
    ap.add_argument("--refine", default="lazy", choices=["lazy", "full"])
    ap.add_argument("--n-class-pairs", type=int, default=None, help="limit class pairs (smoke tests)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = get_profile(args.profile)
    device = cfg.device
    arch = next((n for n in cfg.evaluator_names if n != "clip_zeroshot"), "resnet18")
    den = EDMDenoiser(cfg.edm_checkpoint_uncond, device=device)
    feat = S.ResnetFeatureExtractor(arch, CHECKPOINTS_DIR / f"{arch}_cifar10.pt", device=device)

    # realism function: feature-kNN to a fixed real bank
    train = CIFAR10Canonical(train=True, download=True)
    bank = S.build_feature_bank(feat, _balanced_real(train, 2000))
    realism_fn = lambda imgs: S.feature_knn_realism(feat, bank, imgs.to(device), k=args.knn_k)

    print("Stage 1: building node set...")
    ns = BG.build_node_set(
        den, anchors_per_class=args.anchors_per_class, n_filler=args.n_filler,
        pairs_per_classpair=args.pairs_per_classpair, n_t=args.n_t,
        interp_sigmas=tuple(float(s) for s in args.interp_sigmas.split(",")),
        include_pixel_linear=not args.no_pixel_linear, decode_steps=args.decode_steps,
        seed=args.seed, device=device, n_class_pairs=args.n_class_pairs)
    print(f"  {ns.images.shape[0]} nodes "
          f"({int((ns.anchor_class>=0).sum())} anchors, "
          f"{sum(p['type']=='filler' for p in ns.provenance)} filler, "
          f"{sum(p['type']=='cross' for p in ns.provenance)} cross, "
          f"{sum(p['type']=='same' for p in ns.provenance)} same)")

    print("Stage 1b: scoring node realism...")
    ns = BG.score_nodes(ns, realism_fn)
    print(f"  R range [{ns.R.min():+.4f}, {ns.R.max():+.4f}] median {np.median(ns.R):+.4f}")

    print("Stage 2: pixel-kNN graph...")
    edges = BG.build_graph(ns, k=args.k, device=device)
    dvals = np.array(list(edges.values()))
    delta = float(np.median(dvals))
    print(f"  {len(edges)} edges, delta (median edge pixel-L2) = {delta:.3f}")

    print("Stage 3: barrier extraction...")
    tau, history, w = BG.barrier_matrix(ns, edges, realism_fn, delta=delta, refine=args.refine)

    # design 5.1: within-class floor; 4.1: route provenance (filler-hub diagnosis)
    within = BG.within_class_floor(ns, w)
    prov = BG.route_provenance(ns, w)
    iu = np.triu_indices(10, 1)
    cross = tau[iu]
    prov_filler = np.mean([c["filler"] for c in prov.values()]) if prov else float("nan")
    prov_own = np.mean([c["own_pair"] for c in prov.values()]) if prov else float("nan")
    prov_other = np.mean([c["other_pair"] for c in prov.values()]) if prov else float("nan")
    print("\ntau* cross-class (upper triangle):")
    print(f"  median {np.nanmedian(cross):+.4f}  range [{np.nanmin(cross):+.4f}, {np.nanmax(cross):+.4f}]")
    print(f"  within-class floor (5.1): median {np.nanmedian(within):+.4f}  range "
          f"[{np.nanmin(within):+.4f}, {np.nanmax(within):+.4f}]")
    print(f"  --> gap (within - cross median): {np.nanmedian(within) - np.nanmedian(cross):+.4f}  "
          f"(design P6: within should EXCEED cross if class basins are resolved)")
    print(f"  route composition (4.1): filler={prov_filler:.2f} own-pair={prov_own:.2f} other-pair={prov_other:.2f}")
    print(f"  node R median (reference realistic level): {np.median(ns.R):+.4f}")

    out_dir = RESULTS_DIR / "barrier" / "tau" / args.profile
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "tau.npy", tau)
    (out_dir / "summary.json").write_text(json.dumps({
        "n_nodes": int(ns.images.shape[0]),
        "R_median": float(np.median(ns.R)),
        "tau_cross_median": float(np.nanmedian(cross)),
        "tau_cross_min": float(np.nanmin(cross)),
        "tau_cross_max": float(np.nanmax(cross)),
        "tau_within_median": float(np.nanmedian(within)),
        "within_minus_cross": float(np.nanmedian(within) - np.nanmedian(cross)),
        "route_filler_frac": float(prov_filler),
        "route_own_pair_frac": float(prov_own),
        "route_other_pair_frac": float(prov_other),
        "n_rounds": len(history),
        "interp_sigmas": args.interp_sigmas,
    }, indent=2))

    # readable matrix
    print("\ntau* matrix (rows/cols = classes):")
    print("      " + " ".join(f"{c[:4]:>5s}" for c in CLASSES))
    for i in range(10):
        row = " ".join(f"{tau[i,j]:+.2f}" if not np.isnan(tau[i,j]) else "  .  " for j in range(10))
        print(f"{CLASSES[i][:5]:>5s} {row}")
    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()
