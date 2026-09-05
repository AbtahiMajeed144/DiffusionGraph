# Graph-Based Realism Barrier — Experimental Results

Companion to `GRAPH_BARRIER_EXPERIMENT.md` (design/definitions; referenced by §). This
file is a precise record of **setup and results** for every experiment run. Section
numbers like §0.3 point to the design doc. All runs 2026-09-04 / 2026-09-05.

---

## 0. Common setup

| Item | Value |
|---|---|
| Diffusion model (all decoding/scoring) | **unconditional** EDM CIFAR-10, `checkpoints/baseline-cifar10-32x32-uncond-vp.pkl`, `EDMDenoiser` (`utils/edm_loader.py`) |
| Conditional model (G5 midpoints, T_σ n/a) | `checkpoints/edm-cifar10-32x32-cond-vp.pkl` (`label_dim=10`) |
| Classifier (manifold in-pair, T_σ, G_confusion) | `resnet50` on 5090 / `resnet18` local, `checkpoints/{arch}_cifar10.pt`, fed canonical `[-1,1]` images |
| Data | `CIFAR10Canonical`; anchors/groups from **test** split (EDM trained on train); feature-kNN bank from **train** split |
| Decode | full reverse ODE (Heun), `num_steps=18` unless noted |
| Envs | 5090 = conda `autoeval` (authoritative n=500/full runs); local = conda `loope` (4 GB card, smoke only; `torch.linalg.qr` in EigenScore fails locally, works on 5090) |
| Code | `diffusiongraph/barrier/{scores,groups,barrier_graph}.py`; `scripts/barrier/{validate_score,inspect_groups,sigma_sweep,manifold_structure,build_tsigma,run_barrier}.py` |

---

## 1. Stage 0 — realism-score gate (§0.1–0.3)

### 1.1 Groups (n = 500 each unless noted), `scripts/barrier/validate_score.py`

- **G1 real** — CIFAR-10 test images (class-balanced).
- **G2 synth** — unconditional EDM samples (full reverse ODE).
- **G3 interp** — slerp-in-noise midpoints, t=0.5, σ=0.5, decoded (`denoise_to_clean`). *Original positive.*
- **G4 degraded**, three subgroups: **G4a** pixel-blend 0.5/0.5 double-exposures (166); **G4b** Gaussian blur σ_blur∈{1,2,3} (166); **G4c** additive noise SNR-matched to G4b (168).
- **G5 cond** — linear class-condition midpoints, c = 0.5·onehot(A)+0.5·onehot(B), sampled from fresh latent via the **conditional** model (`--with-g5`). *Repaired positive.*
- graded-blur set σ_blur∈{0,1,2,3} for the monotonicity criterion.

### 1.2 Criteria (§0.3)

1. AUROC(G1∪G2 vs G4) ≥ 0.90 (far). 2. **AUROC(positive vs G4a) ≥ 0.75 (decisive).** 3. median ordering G1≈G2 > positive > G4. 4. monotone R across graded blur.

### 1.3 Results — original G3 positive, n=500

| Candidate | far AUROC | decisive AUROC (G3 vs G4a) | order | monotone | notes |
|---|---|---|---|---|---|
| SCOPED (arXiv:2510.01456) | — | **0.37** | — | — | ±1 ratio collapse; degenerate |
| Raw score-norm (neg. control) | 0.70* | **0.56** | — | — | fails as predicted |
| Feature-kNN resnet50/CLIP (Sun 2022) | 0.92 (far) | **~0.56** | — | ✓ | near-OOD blind |
| EigenScore (arXiv:2510.07206) | 0.8491 / 0.8497 | **0.7408 / 0.7229** (two runs) | ✓ | ✓ | best; short of 0.75 |
| Graham (arXiv:2211.07740) | 0.8597 | **0.3261** (inverted) | True | False | medians G1/G2/G3/G4 = −0.2353/−0.1446/−1.147/−1.3679 |

*score-norm far from n=20 smoke; all decisive/other n=500 values are 5090 runs. No candidate clears decisive 0.75. Gate FAILED on the G3 positive.

### 1.4 Results — repaired G5 positive, n=500 (`--with-g5`, `--eig-sigmas 0.2,0.3,0.5`, `--graham-sigmas 0.3,0.6,1.3,2.5`)

| candidate | far (≥.90) | **decisive G5 vs G4a (≥.75)** | order | monotone | median G1/G2/G5/G4 |
|---|---|---|---|---|---|
| **EigenScore** | 0.833 | **0.874** | ✓ | ✓ | −1.3452 / −0.6209 / −1.9339 / −4.6385 |
| Graham | 0.868 | 0.682 | ✓ | ✗ | −0.1725 / −0.1286 / −0.396 / −1.5255 |

Per-subgroup diagnostic AUROCs (same run):

| candidate | G3·G4a | G3·G4b | G3·G4c | G5·G4a | G5·G4b | G5·G4c | G2·G4a |
|---|---|---|---|---|---|---|---|
| EigenScore | 0.740 | 0.414 | 0.596 | **0.874** | 0.639 | 0.776 | 0.936 |
| Graham | 0.351 | 0.499 | 0.992 | 0.682 | 0.747 | 0.998 | 0.796 |

EigenScore's far miss (0.833 < 0.90) localizes to **G4b (blur)**: G5·G4b = 0.639, G3·G4b = 0.414. Its mixture separation (G5·G4a = 0.874, G2·G4a = 0.936) passes decisive.

---

## 2. Group inspection (visual), `scripts/barrier/inspect_groups.py`

10 cross-class pairs, seed 0; montage rows = G1_A / G1_B real / G3 slerp-mid (σ=0.5) / G4a pixel-blend / G5 cond-mid. Output `results/barrier/stage0/local_poc/group_inspection.png`. Observed: G3 slerp midpoints at σ=0.5 are frequently off-manifold smears (several columns visibly degraded/noise); G4a are recognizable double-exposures with sharp real texture; G5 are coherent single objects. (n=10 visual, superseded by §4 quantitative measurement.)

---

## 3. σ-sweep, `scripts/barrier/sigma_sweep.py`

Same 10 pairs; slerp midpoint decoded at σ ∈ {0.5, 1, 2, 4, 8}. Output `results/barrier/stage0/local_poc/sigma_sweep.png`. Observed: decoded-midpoint realism rises monotonically with σ; by σ=8 the reverse ODE emits coherent single-class images unrelated to the pair (SNR at σ=8 is small, output ≈ unconditional sample). Quantified in §4.

---

## 4. Manifold structure, `scripts/barrier/manifold_structure.py`, n=500 (resnet50, 5090)

`--n 500 --sigmas 0.5,1.0,1.5,2.0,3.0,5.0,8.0`. Two probes:
- **realism** = feature-kNN to a fixed 2000-image real train bank (k=5); higher = on-manifold.
- **in-pair** = frac(classifier argmax ∈ {A,B}); classifier posterior *balance* dropped (overconfident, `maxp≈1` even on pixel blends — reported as caveated secondary only).

| group | realism | in-pair |
|---|---|---|
| slerp s=0.5 | −0.0733 | 0.60 |
| slerp s=1.0 | −0.0613 | 0.57 |
| slerp s=1.5 | −0.0485 | 0.57 |
| slerp s=2.0 | −0.0500 | 0.48 |
| slerp s=3.0 | −0.0443 | 0.43 |
| slerp s=5.0 | −0.0408 | 0.34 |
| slerp s=8.0 | −0.0406 | 0.27 |
| **G1 real** | **−0.0411** | **0.92** |
| G2 synth | −0.0419 | — |
| G4a blend | −0.0817 | 0.78 |

Realism rises and in-pair falls with σ; real images (0.92, −0.0411) lie beyond the whole midpoint trajectory (max midpoint in-pair 0.60, at the least-realistic σ=0.5). Output `results/barrier/manifold/rtx5090/{structure.json, realism_vs_inpair.png}`. (Local n=150 resnet18 run reproduced the same shape.)

---

## 5. T_σ / P5, `scripts/barrier/build_tsigma.py`, per-class 200 (5090)

`--per-class 200 --sigmas 0.25,0.5,1.0,1.5,2.0,3.0 --steps 18`. T_σ[A,B] = P(classifier(denoise(A + σ·ε)) = B), uncond model. Compared to saved τ* (§6) via Spearman on symmetrized T_σ, 45 off-diagonal.

| σ | class-retention (diag) | Spearman(τ*, T_sym) |
|---|---|---|
| 0.25 | 0.91 | −0.347 |
| 0.5 | 0.87 | −0.251 |
| 1.0 | 0.73 | −0.238 |
| 1.5 | 0.63 | −0.221 |
| 2.0 | 0.55 | −0.135 |
| 3.0 | 0.42 | +0.006 |

Retention decreases smoothly (transition crossing ~0.5 near σ≈2.5); max |Spearman(τ*, T_sym)| = 0.35. Output `results/barrier/tsigma/rtx5090/{Tsigma_s*.npy, p5.json}`.

---

## 6. Barrier τ* — Stages 1–3 + controls, `scripts/barrier/run_barrier.py` (5090)

Verdict run: `--anchors-per-class 24 --n-filler 1500 --pairs-per-classpair 6 --n-t 13 --interp-sigmas 0.5,2.0,8.0 --k 20 --refine lazy --controls --compare --confirm-eigenscore --n-perm 200`.

### 6.1 Node set (§1.1) — 16,260 nodes

| type | count | construction |
|---|---|---|
| anchors | 240 | 24 real test / class |
| filler | 1,500 | unconditional samples |
| cross-interp | 11,880 | 45 pairs × 6 endpoint-pairs × [3 σ (0.5,2,8) × 11 interior-t slerp + 11 pixel-linear] |
| same-interp | 2,640 | 10 classes × same construction (within-class null) |

Node realism (feature-kNN, k=5, 2000-image train bank): range [−0.2309, −0.0078], median −0.0476.

### 6.2 Graph + weights (§2)

Pixel-L2 symmetric kNN, k=20, + MST union → 241,312 edges; δ (median edge pixel-L2) = 15.041. Edge weight = min R over segment interior points (§2.2); lazy path refinement (§2.3), 6 rounds (path-edge counts 171→190, weights still changing at cutoff). Realism for structure = feature-kNN; EigenScore used only to re-confirm bottleneck paths (§6.4).

### 6.3 τ* (§3)

Cross-class (45 off-diagonal): **median −0.0260, range [−0.0317, −0.0164]**. Node R median (reference) −0.0476.

```
       plan  auto  bird   cat  deer   dog  frog  hors  ship  truc
plane   .   -0.02 -0.03 -0.03 -0.02 -0.03 -0.02 -0.02 -0.02 -0.02
 auto -0.02   .   -0.03 -0.03 -0.02 -0.03 -0.02 -0.02 -0.02 -0.02
 bird -0.03 -0.03   .   -0.03 -0.03 -0.03 -0.03 -0.03 -0.03 -0.03
  cat -0.03 -0.03 -0.03   .   -0.03 -0.03 -0.03 -0.03 -0.03 -0.03
 deer -0.02 -0.02 -0.03 -0.03   .   -0.03 -0.02 -0.02 -0.02 -0.02
  dog -0.03 -0.03 -0.03 -0.03 -0.03   .   -0.03 -0.03 -0.03 -0.03
 frog -0.02 -0.02 -0.03 -0.03 -0.02 -0.03   .   -0.02 -0.02 -0.02
horse -0.02 -0.02 -0.03 -0.03 -0.02 -0.03 -0.02   .   -0.02 -0.02
 ship -0.02 -0.02 -0.03 -0.03 -0.02 -0.03 -0.02 -0.02   .   -0.02
truck -0.02 -0.02 -0.03 -0.03 -0.02 -0.03 -0.02 -0.02 -0.02   .
```

Structure present: bird/cat/dog rows at −0.03, remaining classes (incl. deer/frog/horse) at −0.02.

### 6.4 Controls, nulls, comparisons

| Test (design §) | Result |
|---|---|
| Route provenance (4.1) | filler 0.25, own-pair 0.04, other-pair 0.67 |
| Within-class floor, restricted (5.1) | median −0.0435, range [−0.1037, −0.0202] |
| Within-class floor, fair (full graph) | median −0.0242; **P6 gap (fair within − cross) = +0.0019** |
| Shuffled-R null, 200 perms (5.2) | real τ* spread IQR = 0.0046; null median 0.0020, p95 0.0044 → **real at 97th percentile**; median \|Spearman\| real-vs-shuffled = 0.238 |
| Filler-removal | cross τ* −0.0260 → −0.0262 (**delta −0.0002**, 0 pairs disconnected) |
| P2: τ* vs G_pixel / G_clip centroid affinity | Spearman **−0.057 / −0.143** |
| EigenScore confirmation (bottleneck paths) | Spearman(feature-kNN τ*, EigenScore path-min) = **+0.548** (45 pairs); EigenScore path-min median −7.686, range [−14.013, −2.647] |

Output `results/barrier/tau/rtx5090/{tau.npy, eig_tau.npy, summary.json, controls.json, compare.json}`.

### 6.5 Prediction scorecard (design §8, thresholds as written there)

| Prediction | Threshold | Value | Met |
|---|---|---|---|
| P1 τ* structure above shuffled-R null | spread > null | 97th percentile | yes |
| P2 τ* not ∝ G_pixel/G_clip | \|ρ\| < 0.8 | 0.057 / 0.143 | yes |
| P3 vehicle/animal subtrees | distinct subtrees | bird/cat/dog anomaly, deer/frog/horse with vehicles | no |
| P4 load-bearing bridge (excision, §4.2) | ≥1 triple | not run (Stage 4.2 not built) | untested |
| P5 τ* ≠ T_σ | ρ < 0.8 | max 0.35 | yes (τ* near-uniform; see §5) |
| P6 within > cross | gap > 0 | +0.0019 (fair) | yes |

Half-scale earlier run (identical config, without `--controls/--compare`, `--n-perm` default): same τ* medians; shuffled-R 96th percentile at 50 perms; EigenScore confirm ρ = 0.269–0.445 across runs (EigenScore Monte-Carlo variance).

### 6.6 Known defects of this τ* run (do not treat §6.3–6.5 as a validated result)

These are established from the numbers above; they mean the τ* matrix in §6.3 carries no
pairwise barrier information and rests on an unvalidated score.

1. **τ* is degenerate: the printed matrix equals `min(f(A),f(B))`** with `f(bird,cat,dog)=−0.03`,
   `f(else)=−0.02` (holds for all 45 entries at 2-decimal precision). This is a per-class
   realism offset propagated through the min — ~10 numbers, not 45, i.e. **no pairwise
   connectivity structure.** Full-precision test (added, `--degeneracy` prints inline):
   Spearman(τ*, `min(f(A),f(B))`) with `f(A)` = median node R over class-A anchors+interpolants.
2. **The shuffled-R null does not discriminate this.** Permuting R destroys the per-class
   offsets, so `min(f(A),f(B))` exceeds the null by construction; the 97th-percentile
   result (p≈0.03, single test) is consistent with a pure per-class offset and no barrier.
   P3-fail and P5-"pass" both follow from τ* being near-constant.
3. **τ* (−0.026) > G1 real (−0.041).** The bottleneck of the worst point on the route
   scores as more on-manifold than real CIFAR test images — impossible for a correct R.
   Max-min optimization selects nodes where feature-kNN over-estimates (near a bank image);
   bottleneck optimization is adversarial against the estimator's max-error regions.
4. **Realism score is unvalidated for this use.** τ* was built on **feature-kNN**, which
   scored ~0.56 decisive on the G3 positive (§1.3, "near-OOD blind") and was **never
   measured on G5**. §4 shows its sign is wrong on the near-OOD pair (slerp −0.0733 > blend
   −0.0817). The EigenScore confirmation ρ = 0.27–0.55 means the two scores measure
   materially different things, and the result-carrying one failed/was-untested at the gate.
5. **P6 is within noise, not a pass.** Fair-within − cross = +0.0019 against IQR 0.0046.
6. **Refinement did not converge** (§6.2): 6 rounds, path-edge counts 171→190, weights
   still changing at the cap. τ* is an unconverged upper bound (refinement only lowers
   weights → true τ* is below reported), and the whole spread is 0.015.

---

## 7. Reproduction

```bash
# Stage 0 gate, G5 repaired positive:
python scripts/barrier/validate_score.py --profile rtx5090 --candidates eigenscore,graham \
    --n-per-group 500 --with-g5 --eig-sigmas 0.2,0.3,0.5 --graham-sigmas 0.3,0.6,1.3,2.5 --graham-steps 18 --graham-batch 256

# Group inspection / sigma-sweep:
python scripts/barrier/inspect_groups.py --profile local_poc --n 10 --seed 0
python scripts/barrier/sigma_sweep.py   --profile local_poc --n 10 --sigmas 0.5,1.0,2.0,4.0,8.0

# Manifold structure (n=500):
python scripts/barrier/manifold_structure.py --profile rtx5090 --n 500 --sigmas 0.5,1.0,1.5,2.0,3.0,5.0,8.0

# T_sigma / P5 (auto-loads tau.npy):
python scripts/barrier/build_tsigma.py --profile rtx5090 --per-class 200 --sigmas 0.25,0.5,1.0,1.5,2.0,3.0 --steps 18

# Barrier tau* + all controls (verdict run):
python scripts/barrier/run_barrier.py --profile rtx5090 --anchors-per-class 24 --n-filler 1500 \
    --pairs-per-classpair 6 --n-t 13 --interp-sigmas 0.5,2.0,8.0 --k 20 --refine lazy \
    --controls --compare --confirm-eigenscore --n-perm 200
```

## 8. Implementation notes (traceable)

- `barrier/scores.py` — SCOPED (component-averaged ratio to avoid ±1 blow-up), raw score-norm, Feature-kNN (resnet/CLIP), EigenScore (forward-only subspace iteration, central-difference JVPs), Graham (`denoise_to_clean` reconstruction error = MSE + perceptual cosine across σ), `hutchinson_divergence` (single-sided tr J, distinct from Frobenius `hutchinson_trace`).
- `barrier/groups.py` — `build_validation_groups` (G1–G4 + graded); `condition_midpoints` / `random_cross_class_pairs` (G5); `_balanced_real` returns exactly n (fixed `n//10` rounding bug that crashed the n=500 G4b loop).
- `barrier/barrier_graph.py` — Stage 1 node set (interpolant decode batched across t; σ-range seeding), pixel-kNN graph + MST, segment-min realism edge weights with lazy path refinement, union-find barrier extraction; `within_class_floor(fair=)`, `route_provenance`, `shuffled_R_null`, `filler_removed_tau`, `comparison_graphs`, `maxmin_over_subgraph`.
- Feature-kNN realism proxy requires a **fixed** ~2000-image bank + small k; an n-scaled bank + large k inverts the real>blend ordering.
- EigenScore `torch.linalg.qr` fails on the local 4 GB card ("GET was unable to find an engine"); runs on the 5090. The `--confirm-eigenscore` block is wrapped non-fatally (τ* is saved before it runs).
- Node/interpolant σ seeded across {0.5, 2, 8} (not the design's single σ=0.5) so the between-region contains realistic candidates as well as off-manifold ones (grounded in §4).
