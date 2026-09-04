"""
GRAPH_BARRIER_EXPERIMENT.md Stage 0 -- the HARD GATE.

Validates realism-score candidates against four groups (G1 real, G2 synthetic,
G3 manifold-aware interpolant midpoints, G4 degraded) and reports the four
pass criteria per candidate. Graham dropped for cost; EigenScore deferred as
the fallback. If no candidate passes criterion 2 (the decisive G3-vs-G4a
separation), STOP -- realism estimation becomes the research problem.

Usage:
    # quick local correctness pass (tiny groups):
    python scripts/barrier/validate_score.py --profile local_poc --n-per-group 20 --candidates scoped,score_norm,clip_knn
    # real run on the 5090:
    python scripts/barrier/validate_score.py --profile rtx5090

Output: results/barrier/stage0/<profile>/  { table.md, metrics.json, hist_<cand>.png }
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diffusiongraph.config import get_profile, RESULTS_DIR, CHECKPOINTS_DIR
from diffusiongraph.utils.edm_loader import EDMDenoiser
from diffusiongraph.barrier.groups import build_validation_groups
from diffusiongraph.barrier import scores as S

DEGRADED = ["G4a_blend", "G4b_blur", "G4c_noise"]


def _score_diffusion(denoiser, images, sigma, num_noise, num_probes, batch, device):
    """Return {'scoped':arr,'score_norm':arr} for a set of images at one sigma."""
    out_scoped, out_norm = [], []
    for i in range(0, images.shape[0], batch):
        x0 = images[i:i + batch].to(device)
        r = S.scoped_and_norm(denoiser, x0, sigma, num_noise=num_noise, num_probes=num_probes)
        out_scoped.append(r["scoped"]); out_norm.append(r["score_norm"])
        if device == "cuda":
            torch.cuda.empty_cache()
    return {"scoped": np.concatenate(out_scoped), "score_norm": np.concatenate(out_norm)}


def _eig_mbar(denoiser, images, sigma, K, n_iter, n_real, c, batch, device):
    """m_bar (sum of top-K posterior-covariance eigenvalues) per image, one sigma."""
    out = []
    for i in range(0, images.shape[0], batch):
        x0 = images[i:i + batch].to(device)
        out.append(S.eigenscore_mbar(denoiser, x0, sigma, K=K, n_iter=n_iter, n_real=n_real, c=c).cpu().numpy())
        if device == "cuda":
            torch.cuda.empty_cache()
    return np.concatenate(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="rtx5090")
    p.add_argument("--n-per-group", type=int, default=500)
    p.add_argument("--sigmas", default="0.1,0.5", help="SCOPED/score-norm noise-eval levels")
    p.add_argument("--candidates", default="scoped,score_norm,resnet_knn,clip_knn,eigenscore",
                   help="subset of: scoped,score_norm,resnet_knn,clip_knn,eigenscore")
    p.add_argument("--num-noise", type=int, default=4)
    p.add_argument("--num-probes", type=int, default=4)
    p.add_argument("--scoped-batch", type=int, default=32)
    p.add_argument("--k", type=int, default=50)
    # EigenScore params (arXiv:2510.07206)
    p.add_argument("--eig-sigmas", default="0.2,0.3,0.5",
                   help="EigenScore noise levels (z-scored + summed). MUST be LOW: the "
                        "OOD->larger-eigenvalue direction is clean at sigma<=0.5 but INVERTS by "
                        "sigma=2.0 (unit-tested); summing across the inversion cancels the signal.")
    p.add_argument("--eig-K", type=int, default=3)
    p.add_argument("--eig-iters", type=int, default=5)
    p.add_argument("--eig-real", type=int, default=2, help="noise realizations averaged")
    p.add_argument("--eig-c", type=float, default=1e-2, help="central-difference step")
    p.add_argument("--eig-batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = get_profile(args.profile)
    device = cfg.device
    sigmas = [float(s) for s in args.sigmas.split(",")]
    want = set(args.candidates.split(","))
    out_dir = RESULTS_DIR / "barrier" / "stage0" / args.profile
    (out_dir).mkdir(parents=True, exist_ok=True)

    print(f"=== Stage 0 score validation | profile={args.profile} n/group={args.n_per_group} sigmas={sigmas} ===")
    denoiser = EDMDenoiser(cfg.edm_checkpoint_uncond, device=device)

    print("Building validation groups...")
    G = build_validation_groups(denoiser, n_per_group=args.n_per_group,
                                sigma_interp=min(sigmas), seed=args.seed, device=device)
    ref_bank_imgs = G.pop("_ref_bank")
    graded = G.pop("graded_blur")
    group_names = ["G1_real", "G2_synth", "G3_interp", "G4a_blend", "G4b_blur", "G4c_noise"]
    for gn in group_names:
        print(f"  {gn}: {tuple(G[gn].shape)}")

    # candidate -> {group_name: R array}, plus graded {sigma_blur: R}
    results = {}

    # --- diffusion candidates (scoped, score_norm) at each sigma ---
    if "scoped" in want or "score_norm" in want:
        for sigma in sigmas:
            per_group = {gn: _score_diffusion(denoiser, G[gn], sigma, args.num_noise, args.num_probes, args.scoped_batch, device) for gn in group_names}
            graded_r = {sb: _score_diffusion(denoiser, graded[sb], sigma, args.num_noise, args.num_probes, args.scoped_batch, device) for sb in graded}
            for cand in ("scoped", "score_norm"):
                if cand in want:
                    name = f"{cand}@s{sigma}"
                    results[name] = {"groups": {gn: per_group[gn][cand] for gn in group_names},
                                     "graded": {sb: graded_r[sb][cand] for sb in graded}}
            print(f"  scored diffusion candidates at sigma={sigma}")

    # --- feature-kNN candidates ---
    def _knn_candidate(extractor, tag):
        bank = S.build_feature_bank(extractor, ref_bank_imgs.to(device) if False else ref_bank_imgs)
        groups_r = {gn: S.feature_knn_realism(extractor, bank, G[gn].to(device), k=args.k) for gn in group_names}
        graded_r = {sb: S.feature_knn_realism(extractor, bank, graded[sb].to(device), k=args.k) for sb in graded}
        results[tag] = {"groups": groups_r, "graded": graded_r}
        print(f"  scored {tag}")

    if "resnet_knn" in want:
        # first non-clip evaluator arch from the profile (resnet50 on 5090, resnet18 local)
        arch = next((n for n in cfg.evaluator_names if n != "clip_zeroshot"), "resnet18")
        ckpt = CHECKPOINTS_DIR / f"{arch}_cifar10.pt"
        if ckpt.exists():
            _knn_candidate(S.ResnetFeatureExtractor(arch, ckpt, device=device), f"{arch}_knn")
        else:
            print(f"  (skipping resnet_knn: {ckpt} not found -- train it first)")

    if "clip_knn" in want:
        from diffusiongraph.models.embeddings import ClipZeroShot
        _knn_candidate(S.ClipFeatureExtractor(ClipZeroShot(device=device)), "clip_knn")

    # --- EigenScore (near-OOD specialist / fallback) ---
    if "eigenscore" in want:
        eig_sigmas = [float(s) for s in args.eig_sigmas.split(",")]
        # per-sigma m_bar for every group/graded set; z-score against the REAL
        # reference bank (real train images, disjoint from G1), then sum over sigma.
        # OOD -> larger eigenvalues -> larger m_bar -> larger S -> realism = -S.
        z_groups = {gn: np.zeros(G[gn].shape[0]) for gn in group_names}
        z_graded = {sb: np.zeros(graded[sb].shape[0]) for sb in graded}
        kw = dict(K=args.eig_K, n_iter=args.eig_iters, n_real=args.eig_real, c=args.eig_c,
                  batch=args.eig_batch, device=device)
        ref_sub = ref_bank_imgs[:min(400, ref_bank_imgs.shape[0])]
        for sigma in eig_sigmas:
            mu = _eig_mbar(denoiser, ref_sub, sigma, **kw)
            m_mu, m_sd = float(np.mean(mu)), float(np.std(mu) + 1e-8)
            for gn in group_names:
                z_groups[gn] += (_eig_mbar(denoiser, G[gn], sigma, **kw) - m_mu) / m_sd
            for sb in graded:
                z_graded[sb] += (_eig_mbar(denoiser, graded[sb], sigma, **kw) - m_mu) / m_sd
            print(f"  scored eigenscore at sigma={sigma}")
        results["eigenscore"] = {"groups": {gn: -z_groups[gn] for gn in group_names},
                                 "graded": {sb: -z_graded[sb] for sb in graded}}

    # --- metrics per candidate ---
    def med(a): return float(np.median(a))
    table = {}
    for cand, r in results.items():
        gr = r["groups"]
        pos_far = np.concatenate([gr["G1_real"], gr["G2_synth"]])
        neg_far = np.concatenate([gr[d] for d in DEGRADED])
        auroc_far = S.auroc(pos_far, neg_far)                          # criterion 1, need >=0.90
        auroc_dec = S.auroc(gr["G3_interp"], gr["G4a_blend"])          # criterion 2 (decisive), need >=0.75
        m1, m2, m3 = med(gr["G1_real"]), med(gr["G2_synth"]), med(neg_far)
        m_g3 = med(gr["G3_interp"])
        ordering_ok = (min(m1, m2) > m_g3 > m3)                        # criterion 3
        graded_meds = [med(r["graded"][sb]) for sb in [0, 1, 2, 3]]
        monotone_ok = all(graded_meds[i] > graded_meds[i + 1] for i in range(3))  # criterion 4
        passes = (auroc_far >= 0.90) and (auroc_dec >= 0.75) and ordering_ok and monotone_ok
        table[cand] = dict(
            auroc_far=round(auroc_far, 4), auroc_decisive=round(auroc_dec, 4),
            median_G1=round(m1, 4), median_G2=round(m2, 4), median_G3=round(m_g3, 4), median_G4=round(m3, 4),
            ordering_ok=bool(ordering_ok), graded_blur_medians=[round(x, 4) for x in graded_meds],
            monotone_ok=bool(monotone_ok), PASSES_ALL=bool(passes),
        )

    # --- histograms ---
    for cand, r in results.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        for gn, color in zip(["G1_real", "G2_synth", "G3_interp", "G4a_blend"], ["C2", "C0", "C1", "C3"]):
            vals = r["groups"][gn]
            vals = vals[np.isfinite(vals)]
            ax.hist(vals, bins=30, alpha=0.5, density=True, label=gn, color=color)
        ax.set_title(f"{cand}  (decisive G3-vs-G4a AUROC={table[cand]['auroc_decisive']})")
        ax.set_xlabel("R (higher = more realistic)"); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out_dir / f"hist_{cand.replace('@','_').replace('.','p')}.png", dpi=140)
        plt.close(fig)

    # --- deliverable table ---
    (out_dir / "metrics.json").write_text(json.dumps(table, indent=2))
    lines = ["# Stage 0 — realism score validation", "",
             f"profile={args.profile}, n/group={args.n_per_group}, sigmas={sigmas}", "",
             "| candidate | AUROC far (>=.90) | **AUROC decisive (>=.75)** | order ok | monotone ok | median G1/G2/G3/G4 | PASS |",
             "|---|---|---|---|---|---|---|"]
    for cand, t in table.items():
        lines.append(f"| {cand} | {t['auroc_far']} | **{t['auroc_decisive']}** | {t['ordering_ok']} | "
                     f"{t['monotone_ok']} | {t['median_G1']}/{t['median_G2']}/{t['median_G3']}/{t['median_G4']} | "
                     f"{'**YES**' if t['PASSES_ALL'] else 'no'} |")
    passers = [c for c, t in table.items() if t["PASSES_ALL"]]
    lines += ["", f"**Gate:** {'PASS — cheapest passing candidate: ' + min(passers) if passers else 'FAIL — no candidate passes all four criteria. Per the design, STOP: realism estimation is the blocker. Consider EigenScore (fallback) or Graham before declaring failure.'}"]
    (out_dir / "table.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()
