"""
Manifold structure of the between-class region (the pivot result).

Headline claim: for EDM-CIFAR a between-class midpoint is REALISTIC XOR MIXED --
the (high-realism, high-betweenness) corner is empty. Measured with two robust,
independently-trusted probes (NOT the near-OOD score that failed Stage 0):

  REALISM axis      feature-kNN realism to a real bank (Stage-0 far-AUROC 0.92),
                    plus classifier max-prob (confidence).
  BETWEENNESS axis  classifier posterior over the 10 classes:
                      mix   = 2*min(p_A, p_B)   (1.0 iff a perfect 50/50 A|B mix;
                                                 ~0 if collapsed to one class OR
                                                 mass sits on a THIRD class)
                      p_AB  = p_A + p_B          (endpoint mass)
                      H     = entropy            (high+low p_AB => off-manifold garbage)

Sweeps slerp midpoints over noise level sigma. References: G1 real endpoints,
G2 unconditional samples, G4a pixel blends. Emits a table + a realism-vs-mix
scatter coloured by sigma, with the reference groups overlaid.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from diffusiongraph.config import get_profile, RESULTS_DIR, CHECKPOINTS_DIR
from diffusiongraph.utils.edm_loader import EDMDenoiser
from diffusiongraph.data.cifar10 import CIFAR10Canonical
from diffusiongraph.paths.slerp_noise import slerp
from diffusiongraph.paths.base import forward_diffuse
from diffusiongraph.barrier.groups import random_cross_class_pairs, _balanced_real
from diffusiongraph.barrier import scores as S


def _load_classifier(arch, device):
    from diffusiongraph.models.classifiers import ARCHITECTURES
    m = ARCHITECTURES[arch]()
    st = torch.load(CHECKPOINTS_DIR / f"{arch}_cifar10.pt", map_location=device)
    m.load_state_dict(st["model_state_dict"] if "model_state_dict" in st else st)
    return m.to(device).eval()


@torch.no_grad()
def _posterior(clf, imgs, device, bs=256):
    out = []
    for i in range(0, imgs.shape[0], bs):
        out.append(F.softmax(clf(imgs[i:i + bs].to(device)), dim=1).cpu())
    return torch.cat(out, 0)


def _probe(post, realism, classes_a=None, classes_b=None):
    """post [N,10] softmax, realism [N]. classes_a/b [N] endpoint labels (None for
    reference groups with no defined pair).

    Betweenness is measured by the ROBUST argmax DECISION, not the posterior
    balance: the CIFAR classifier is overconfident (maxp~1 even on a literal 50/50
    pixel blend), so per-image mass on {A,B} is meaningless. What is trustworthy is
    WHICH class the midpoint resolves to:
      in_pair_frac = frac(argmax in {A,B})   -- stayed in the pair's neighborhood
      third_frac   = 1 - in_pair_frac        -- wandered to an unrelated basin
    p_AB/mix/entropy are kept only as caveated secondary numbers."""
    realism = np.asarray(realism)
    N = post.shape[0]
    maxp = post.max(1).values.numpy()
    ent = (-(post * (post + 1e-12).log()).sum(1)).numpy()
    d = {"realism_med": float(np.median(realism)), "maxprob_med": float(np.median(maxp)),
         "entropy_med": float(np.median(ent)), "_realism": realism}
    if classes_a is not None:
        pa = post[torch.arange(N), classes_a].numpy()
        pb = post[torch.arange(N), classes_b].numpy()
        am = post.argmax(1).numpy()
        in_pair = ((am == np.asarray(classes_a)) | (am == np.asarray(classes_b)))
        d.update(in_pair_frac=float(in_pair.mean()), third_class_frac=float((~in_pair).mean()),
                 p_ab_med=float(np.median(pa + pb)), mix_med_CAVEAT=float(np.median(2 * np.minimum(pa, pb))))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="local_poc")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--sigmas", default="0.5,1.0,2.0,4.0,8.0")
    ap.add_argument("--steps", type=int, default=18)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = get_profile(args.profile)
    device = cfg.device
    arch = next((n for n in cfg.evaluator_names if n != "clip_zeroshot"), "resnet18")
    den = EDMDenoiser(cfg.edm_checkpoint_uncond, device=device)
    clf = _load_classifier(arch, device)
    feat = S.ResnetFeatureExtractor(arch, CHECKPOINTS_DIR / f"{arch}_cifar10.pt", device=device)
    test = CIFAR10Canonical(train=False, download=True)
    train = CIFAR10Canonical(train=True, download=True)
    sweep = [float(s) for s in args.sigmas.split(",")]

    # real reference bank for feature-kNN realism (disjoint train split).
    # FIXED size (not n-scaled): a small bank + large k makes the k-th-NN distance
    # track class structure rather than realism and inverts the ordering.
    bank_imgs = _balanced_real(train, 2000)
    bank = S.build_feature_bank(feat, bank_imgs)
    realism = lambda imgs: S.feature_knn_realism(feat, bank, imgs.to(device), k=args.k)

    # n cross-class real pairs (shared across the whole sweep)
    pairs = random_cross_class_pairs(args.n, seed=args.seed)
    ca = torch.tensor([p[0] for p in pairs]); cb = torch.tensor([p[1] for p in pairs])
    g = torch.Generator().manual_seed(args.seed)
    a_imgs, b_imgs = [], []
    for (pa, pb) in pairs:
        ia = test.indices_for_class(pa)[int(torch.randint(len(test.indices_for_class(pa)), (1,), generator=g))]
        ib = test.indices_for_class(pb)[int(torch.randint(len(test.indices_for_class(pb)), (1,), generator=g))]
        a_imgs.append(test[ia][0]); b_imgs.append(test[ib][0])
    xa = torch.stack(a_imgs); xb = torch.stack(b_imgs)

    rows = {}
    traj = []  # (sigma, in_pair_frac, realism_med, realism_array) for the trajectory
    for sig in sweep:
        mids = []
        for i in range(0, args.n, 128):
            a_s = forward_diffuse(xa[i:i+128].to(device), sig, generator=g)
            b_s = forward_diffuse(xb[i:i+128].to(device), sig, generator=g)
            mids.append(den.denoise_to_clean(slerp(a_s, b_s, 0.5), sig,
                        class_labels=None, num_steps=args.steps).cpu())
        mid = torch.cat(mids, 0)
        d = _probe(_posterior(clf, mid, device), realism(mid), ca, cb)
        traj.append((sig, d["in_pair_frac"], d["realism_med"], d.pop("_realism")))
        rows[f"slerp@s{sig:g}"] = d
        print(f"  slerp s={sig:g}: realism={d['realism_med']:+.4f} in_pair={d['in_pair_frac']:.2f} "
              f"third={d['third_class_frac']:.2f} maxp={d['maxprob_med']:.3f}")

    # references
    for tag, imgs, la, lb in [("G1_real", xa, ca, cb),
                              ("G4a_blend", (0.5 * xa + 0.5 * xb).clamp(-1, 1), ca, cb)]:
        d = _probe(_posterior(clf, imgs, device), realism(imgs), la, lb)
        d.pop("_realism", None)
        rows[tag] = d
    g2 = den.sample(class_idx=None, batch_size=min(args.n, 200), num_steps=args.steps, seed=args.seed).cpu()
    d = _probe(_posterior(clf, g2, device), realism(g2)); d.pop("_realism", None)
    rows["G2_synth"] = d

    out_dir = RESULTS_DIR / "barrier" / "manifold" / args.profile
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "structure.json").write_text(json.dumps(rows, indent=2))

    # trajectory: as sigma sweeps, does (in_pair, realism) ever reach TOP-RIGHT?
    fig, ax = plt.subplots(figsize=(7.5, 6))
    xs = [t[1] for t in traj]; ys = [t[2] for t in traj]
    ax.plot(xs, ys, "-o", color="k", zorder=3)
    for sig, ip, rm, _ in traj:
        ax.annotate(f"s={sig:g}", (ip, rm), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.axhline(rows["G1_real"]["realism_med"], ls="--", color="C2", lw=1, label="G1 real realism")
    ax.axhline(rows["G2_synth"]["realism_med"], ls="--", color="C0", lw=1, label="G2 synth realism")
    ax.axhline(rows["G4a_blend"]["realism_med"], ls=":", color="C3", lw=1, label="G4a blend realism")
    ax.axvline(rows["G1_real"]["in_pair_frac"], ls="--", color="C2", lw=0.8, alpha=0.6)
    ax.set_xlabel("semantic locality to pair:  frac(argmax in {A,B})  -> 1 = stayed between the two classes")
    ax.set_ylabel("realism  (feature-kNN to real bank; higher = on-manifold)")
    ax.set_title("Between-class midpoints: realistic XOR in-pair\n"
                 "(TOP-RIGHT = realistic AND between; claim = the trajectory never gets there)")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout(); fig.savefig(out_dir / "realism_vs_inpair.png", dpi=140); plt.close(fig)

    print("\nReferences:")
    for t in ["G1_real", "G2_synth", "G4a_blend"]:
        r = rows[t]
        print(f"  {t:10s}: realism={r['realism_med']:+.4f} maxp={r['maxprob_med']:.3f}"
              + (f" in_pair={r['in_pair_frac']:.2f}" if 'in_pair_frac' in r else ""))
    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()
