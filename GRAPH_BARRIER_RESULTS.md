# Graph-Based Realism Barrier — Experimental Results

Companion to `GRAPH_BARRIER_EXPERIMENT.md`. This file records outcomes only; the
design, definitions, and pass criteria live in that document and are referenced by
section number (e.g. §0.3) throughout.

**Status as of 2026-09-05: STAGE 0 GATE — FAILED.**
No realism score cleared the decisive criterion (§0.3 #2). Per the design's own stop
rule (§0.3 "Gate", "Order of operations" step 1, and "What can go wrong" → *"Stage 0
fails criterion 2 for every candidate"*), **Stages 1–8 were not built.** The blocker is
realism estimation for near-OOD generative interpolants — which is itself the reportable
finding the design anticipated.

---

## 1. What was run

Stage 0 only (§0.1–0.3): validate a realism score `R` before constructing any graph.

- **Groups (§0.1):** G1 real (CIFAR-10 *test*), G2 unconditional EDM samples, G3 slerp
  interpolant midpoints (σ=0.5, decoded through the full reverse ODE), G4 degraded =
  G4a pixel-blend double-exposures / G4b Gaussian blur / G4c SNR-matched noise. Plus a
  graded-blur set σ_blur ∈ {0,1,2,3} for the monotonicity criterion.
- **Model:** unconditional EDM CIFAR-10 (`baseline-cifar10-32x32-uncond-vp.pkl`),
  `EDMDenoiser.denoise_to_clean` for all decoding.
- **Scale:** n = 500 per group on the RTX 5090 (`--profile rtx5090`), the authoritative
  runs. A local n = 20 smoke test (`--profile local_poc`) was used only to verify the
  code path executes end-to-end.

The six candidate scores of §0.2 were implemented in `diffusiongraph/barrier/scores.py`
and driven by `scripts/barrier/validate_score.py`.

## 2. Pass criteria (from §0.3)

1. **AUROC(G1∪G2 vs G4) ≥ 0.90** — easy far separation.
2. **AUROC(G3 vs G4a) ≥ 0.75** — *the decisive test.* Both are class-mixed midpoints;
   one is manifold-aware, one is a literal double exposure. A score that cannot separate
   them is measuring "is this image mixed," not realism.
3. **Median ordering** G1 ≈ G2 > G3 > G4, no inversions.
4. **Monotone** response to graded blur.

The gate is criterion 2. Everything downstream is a monotone function of `R`, so a score
that fails #2 produces a confidently wrong tree.

---

## 3. Results

### 3.1 Headline — the decisive criterion (AUROC G3 vs G4a, need ≥ 0.75)

| Candidate | Family | Decisive AUROC | Verdict |
|---|---|---|---|
| SCOPED (arXiv:2510.01456) | score geometry (‖s‖²·sign / −tr J) | **0.37** | fail (degenerate; ±1 collapse) |
| Raw score-norm | ‖D−x‖/σ² (**negative control**) | **0.56** | fail (as predicted) |
| Feature-kNN, resnet50/CLIP (Sun 2022) | perceptual kNN | **~0.56** | fail (near-OOD blind) |
| **EigenScore (arXiv:2510.07206)** | posterior-covariance top-K spectrum | **0.72 – 0.74** | fail (best; still short of 0.75) |
| Graham (arXiv:2211.07740) | reconstruction error across σ | **0.33** | fail (inverted, < 0.5) |

**No candidate reaches 0.75.** EigenScore is the ceiling at ~0.73 and is stable under
tuning (two independent n=500 runs: 0.7408 and 0.7229), i.e. the ~0.73 is a real ceiling,
not Monte-Carlo noise that a re-run would lift over the bar.

### 3.2 Full n=500 rows where captured

**Graham** (`--graham-sigmas 0.3,0.6,1.3,2.5 --graham-steps 18`, n=500):

| far AUROC | decisive AUROC | order ok | monotone ok | median G1/G2/G3/G4 |
|---|---|---|---|---|
| 0.8597 | **0.3261** | True | False | −0.2353 / −0.1446 / −1.147 / −1.3679 |

**EigenScore** (two n=500 runs):

| run | far AUROC | decisive AUROC |
|---|---|---|
| 1 | 0.8491 | 0.7408 |
| 2 | 0.8497 | 0.7229 |

EigenScore also passed ordering and monotonicity; it fails **only** on the decisive
threshold and (marginally) on far (< 0.90). It is the sole candidate that is coherent in
every respect except the one that matters.

*Provenance:* Graham and EigenScore rows are the RTX-5090 n=500 console outputs. SCOPED's
0.37 and Feature-kNN's ~0.56 decisive are from the same n=500 gate campaign; the raw
score-norm 0.56 is the negative control. The local `results/barrier/stage0/local_poc/`
directory holds only the last n=20 smoke run (overwritten each invocation) and is **not**
the basis for any number above.

---

## 4. What the numbers mean

**Criterion 2 is the wall, and it is the same wall for every family.** Five *independent*
estimator families — score-geometry (SCOPED), raw norm, perceptual-feature kNN,
posterior-covariance spectrum (EigenScore), and reconstruction error (Graham) — all fail
to separate manifold-aware decoded interpolants (G3) from pixel double-exposures (G4a).
They fail on far-different principles, which makes the failure a property of the *problem*
(near-OOD realism of generative interpolants), not of any one implementation.

**Graham's failure is an inversion, and it is instructive.** Its median ordering is
nominally correct (G3 = −1.147 sits above the pooled degraded median G4 = −1.368), yet its
decisive AUROC is **0.33 < 0.5**: on the G3-vs-G4a pair specifically, pixel blends score
*more realistic* than decoded midpoints. Two sharp real images averaged in pixel space
retain real texture that reconstructs well; a slerp-decoded midpoint is a smooth
off-manifold image that reconstructs *worse*. Reconstruction error therefore rewards
exactly the wrong thing on the decisive pair.

**SCOPED is additionally degenerate.** Averaging its ratio statistic collapses toward
sign(·)·1 (the ±1 collapse), so the decisive statistic degrades to a near-coin-flip
(0.37). Flagged honestly as possibly implementation-degenerate, but it does not clear the
gate under the corrected component-averaged form either.

**EigenScore's ~0.73 ceiling is the informative near-miss.** It is the only score that is
otherwise well-behaved, and it plateaus just under the bar. The most likely reading — and
it is consistent with the Phase-1 KILL — is that decoded slerp midpoints are *genuinely*
close in realism to pixel blends at this σ, so no score cleanly separates them because the
separation the criterion demands may not exist at the population level.

---

## 5. Relation to Phase 1 (kept separate, per §0.4)

The design insists the two results not share a sentence, and that independence is honored
here: Phase 1's classifier-statistic (`C`) KILL does **not** logically entail this Stage-0
outcome, and this outcome does not resurrect any Phase-1 routing claim. What can be said
without conflating them is that **both point at the same missing capability**: Phase 1
found that a *classifier* statistic cannot see realism, and Stage 0 finds that no
available *realism* statistic separates near-OOD generative interpolants from degraded
mixtures. The graph machinery (Stages 1–8) was never the bottleneck; realism estimation
is. That convergence is the paper.

---

## 6. Decision

Per §0.3 ("Gate") and the "Order of operations" step 1 — *do not proceed on a failed
score* — the program **stops at Stage 0**. This is the outcome the design pre-registered
under "What can go wrong" → *"Stage 0 fails criterion 2 for every candidate … it would
make realism estimation for near-OOD generative interpolants the actual research
contribution — a narrower but real paper."*

**We do not build Stages 1–8** on EigenScore (or any candidate), because a score that
fails criterion 2 produces a confidently wrong barrier tree — the exact failure the gate
exists to prevent.

### Open routes (if the program is revisited)

1. **Narrow the target paper** to the demonstrated result: *near-OOD realism estimation
   for generative interpolants is unsolved by five standard estimator families*, with the
   G3-vs-G4a decisive test as the benchmark and EigenScore's ~0.73 as the current ceiling.
2. **Change the object, not the estimator.** If a future score genuinely clears 0.75 on
   the G3/G4a decisive test at n≥500 with stable re-runs, Stage 0 re-opens and Stages 1–8
   become buildable unchanged.
3. Lowering the 0.75 threshold is **not** a route — it was set (§0.3) as the point below
   which `R` is measuring "is this image mixed" rather than realism.

---

## 7. Reproduction

Environment: `autoeval` (5090) / `loope` (local). No new libraries were installed on the
5090 per the operating constraint.

```bash
# EigenScore (best candidate, ~0.73 decisive):
python scripts/barrier/validate_score.py --profile rtx5090 --candidates eigenscore \
    --n-per-group 500 --eig-sigmas 0.2,0.3,0.5

# Graham (reserved score, inverted decisive):
python scripts/barrier/validate_score.py --profile rtx5090 --candidates graham \
    --n-per-group 500 --graham-sigmas 0.3,0.6,1.3,2.5 --graham-steps 18 --graham-batch 256

# Full sweep (all six candidates in one pass):
python scripts/barrier/validate_score.py --profile rtx5090 \
    --candidates scoped,score_norm,resnet_knn,clip_knn,eigenscore,graham --n-per-group 500
```

Each run writes `results/barrier/stage0/<profile>/{table.md, metrics.json}` and
per-candidate score-distribution histograms. Note that the header line reports the
diffusion-score `--sigmas` and, since commit `e0c764b`, also the per-candidate
`--graham-sigmas` / `--eig-sigmas` actually used.

## 8. Implementation notes (traceable)

- `diffusiongraph/barrier/scores.py` — SCOPED (component-averaged ratio to avoid ±1
  blow-up), raw score-norm, Feature-kNN (resnet/CLIP), EigenScore (forward-only subspace
  iteration, central-difference JVPs), Graham (`denoise_to_clean` reconstruction error =
  MSE + perceptual cosine across σ), plus a correct single-sided Hutchinson
  `hutchinson_divergence` (tr J), distinct from the existing Frobenius `hutchinson_trace`.
- `diffusiongraph/barrier/groups.py` — `build_validation_groups`; `_balanced_real`
  returns exactly n class-balanced images (a `n//10` rounding bug crashed the n=500 G4b
  loop and was fixed).
- Graham direction was unit-checked before the campaign: real (+0.03) > blend (−0.33) >
  noise (−8.09), all finite — the score is oriented correctly; it simply cannot resolve
  the decisive near-OOD pair.
