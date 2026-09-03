# Phase 1 Gate — Experiment Report

**Status as of this report: KILL on the C-based routing claim (RQ2/RQ3); a PIVOT to RQ4/realism is the only surviving positive route and is not yet measured (§5.10, §7).** The A↔A same-class null, now complete for all three paths (mask-matched, n=90/σ), delivers the decisive result: floor-subtracted cross-class excess is linear +0.148 ≫ slerp +0.043 ≈ tangential +0.037 — the geometry-aware path shows the *least* cross-class activation, so the soft "geometry beats baselines" bar is not just unsupported but reversed. C, however, cannot distinguish realistic third-class members from classifier-fooling chimeras, so whether the geometry path is more *realistic* than condition-interpolation (RQ4) is the one open question that could yield a positive result — it requires a realism metric + visual inspection, not done yet. Strict routing bar is a clean, threshold-robust null; σ=2.0 real C equals the within-class floor exactly. Recommendation: stop the sweep, run the realism analysis (decisive), then write up as either an RQ4 PIVOT or a characterized negative result. Does not depend on the (confirmed-broken, §5.9) permutation control. This document is the running record of every experiment, calibration, and finding.

See `SEED_semantic_class_graph.md` for the research design this implements, `THIRD_PARTY.md` for reference-repo provenance, and `experiment/README.md` for how to reproduce/resume the run itself. This file is the *findings* record; those are the *how-to* records.

---

## 1. What's being tested

SEED §3: does a realism-maximizing geometric path between two CIFAR-10 classes route through a third class more than chance, more than naive baselines, and does that survive a label-permutation control? Measured via:

```
C(A,B) = max over t of [ max over c ∉ {A,B} of p(c | γ(t)) ]
```
computed independently by 3 evaluators, required to hold across all evaluators *and* all seeds before counting as a confirmed routing event (SEED §3.3).

**Decision framework (SEED §3.5):**
- **GO** — routing events reproducible across evaluators/seeds, geometry-aware path beats baselines, permutation control clean → Phase 2.
- **PIVOT** — permutation clean, but routing weak/no clear geometry-vs-baseline separation → reframe around RQ4 (which path objective minimizes chimeras).
- **KILL** — nothing separates path types, or routing survives permutation (artifact) → stop.

Our own code checks two different bars (this distinction turned out to matter a lot — see §5):
- **Soft bar**: average routing strength, geometry path vs. baselines (what `analysis/controls.py`'s GO/PIVOT/KILL logic actually uses).
- **Strict bar**: an individual pair's C(A,B) > τ=0.5 on every evaluator and seed (what "confirmed routing event" means).

---

## 2. Infrastructure built

| Component | What | Key file(s) |
|---|---|---|
| Generator | Pretrained EDM CIFAR-10, **two checkpoints** — conditional (path 1 only) and unconditional (paths 2-4) — a structural fix against CFG/conditioning contamination, not just pinning guidance weight | `utils/edm_loader.py`, `scripts/download_edm_checkpoint.py` |
| Paths | 4 SEED §3.2 path types: linear conditioning, slerp-in-noise, tangential geodesic (JVP-via-double-backward score-Jacobian metric, original implementation — no code exists for Saito–Matsubara), string method (deferred) | `paths/` |
| Evaluators | 3 independent (2 CNN/ViT architectures + CLIP zero-shot), real-label + permuted-label variants for the identifiability control | `models/classifiers.py`, `models/embeddings.py`, `eval/evaluators.py` |
| Data | CIFAR-10 in canonical `[-1,1]` format, sample-pair (not centroid) retrieval per SEED §5, label-permutation dataset | `data/cifar10.py` |
| Gate orchestrator | Sweeps pairs × paths × σ × seeds, computes routing matrices, runs the permutation control, writes decision memo + figures | `scripts/run_gate.py` |
| **Resume cache** | Every combo persisted to disk immediately on completion; interrupting the ~2-day sweep loses at most one in-flight combo | `utils/checkpoint_cache.py` |
| **Post-hoc analysis** | Reads the cache directly (works on partial runs) for relative-dominance and realism metrics beyond the strict threshold | `scripts/analyze_cache.py` |

Full commit history: 15 commits, `2c87adf` (initial) through `35caa40` (latest).

---

## 3. Environments

| | Local dev machine | Remote (Phase 1 real run) |
|---|---|---|
| GPU | RTX 3050 Laptop, 4GB VRAM | RTX 5090, ~32GB VRAM |
| Conda env | `loope` | `autoeval` |
| Role | Pipeline build/debug, small-scale PoC, calibration | The actual 45-pair exhaustive sweep |

**Network issues hit and resolved, worth remembering if repeated on new hardware:**
- torchvision's default CIFAR-10 source (`cs.toronto.edu`) measured <0.2 MB/s — switched to a HuggingFace Hub mirror.
- Naive single-stream downloads (even from HF) throttled hard after ~30MB on the dev network — fixed with a multi-connection Range-request downloader (`scripts/parallel_download.py`), used by default for CIFAR-10.
- `chenyaofo/pytorch-cifar-models`' GitHub-Releases-hosted checkpoint was outright unreachable (DNS/TLS fine, then no data) — abandoned that pretrained-ResNet path in favor of training `resnet50`/`vit_base` from scratch.

---

## 4. Evaluators

| Evaluator | Arch (local, 4GB) | Acc | Arch (5090, real run) | Acc |
|---|---|---|---|---|
| CNN | `resnet18` | 85.9% (89.2% with augmentation) | `resnet50` | **>93%** (reported by user) |
| Transformer | `vit_small` | 61.1% | `vit_base` | **70%** |
| Zero-shot | `clip_zeroshot` (ViT-B/32) | 87.8% | same | not re-validated on 5090 |

**Open concern, not yet resolved:** `vit_base`'s 70% is lower than expected — the reference repo (`omihub777/ViT-CIFAR`) reports >90% from-scratch with heavier augmentation/longer training, and it's a much larger model (~21M params) than the 61%-scoring `vit_small` (~2-3M). We do not have its train-vs-test accuracy curve to know if it's underfit (needs more epochs) or something else — **flagged for follow-up, not investigated**.

**Augmentation:** added (`CIFAR10Canonical(augment=True)`, random 4px-pad crop + horizontal flip) after `resnet18`'s first training run showed clear overfitting (100% train acc by epoch 13, ~85% test acc — a real generalization gap). Used for all evaluator training on both machines since.

---

## 5. Experiment log

### 5.1 `local_poc` (RTX 3050) — first real signal, 4 pairs

Settings: 4 pairs (cat↔dog, dog↔horse, auto↔truck, cat↔airplane), 2 seeds, 1 σ (2.0), 60 optimizer steps, 8 control points, `resnet18`/`vit_small`/`clip_zeroshot`.

**Result (from `results/gate/phase1_gate_poc_local/decision_memo.json`):**

| Path | Mean strength |
|---|---|
| `linear_condition` | 0.0905 |
| `slerp_noise` | 0.2471 |
| `tangential_geodesic` | 0.2934 |

**Decision: GO** (soft bar — geometry beats both baselines, permutation clean, 0.0% false-positive rate).

**But under the strict bar: 0/8 (pair×seed) combinations confirmed.** Every individual value clustered near τ=0.5 rather than clearing it — closest calls were dog↔horse (0.497/0.484/0.426) and cat↔airplane (0.500/0.495/0.443). This first exposed the soft-bar/strict-bar gap that turned out to matter throughout.

### 5.2 5090 `local_smoke` test — later shown non-representative

Single pair (cat↔dog), 1 seed, σ=2.0, `samples_per_class=2` (vs. 16 in the real sweep). Real: linear=0.003/0.49/0.026, slerp=**0.50/0.499/0.413**, tangential=0.50/0.506/0.556. Permuted: clean (0.505/0.500 for tangential).

**This run's `slerp_noise` numbers (0.41-0.50) were later cited as if representative of the full sweep and turned out not to be** — `samples_per_class=2` produces a noisy, small-sample average that spikes easily. The `local_poc` run's `slerp_noise=0.247` (at `samples_per_class=4`, closer to the real sweep's 16) was the better predictor; the real sweep is now showing `slerp_noise` "barely crossing 0.3," consistent with `local_poc`, not the smoke test. **Lesson: don't extrapolate routing-strength numbers from `samples_per_class≤2` runs.**

### 5.3 `convergence_check` — real optimizer-steps calibration

300-step run (cat↔dog, σ=2.0, small scale) to measure the actual curve-energy plateau instead of guessing.

| step | energy | % of total (1→300) decrease captured |
|---|---|---|
| 30 | 483.10 | 49% |
| 90 | 443.35 | 82% |
| **150** | **432.24** | **92%** |
| 300 | 422.61 | 100% (by definition of the window) |

`geodesic_optimizer_steps=150` locked in for `rtx5090` — past the knee, 92% captured for half the compute of 300. **Caveat carried forward, not re-verified**: this was measured at σ=2.0 only; low-σ (0.5, the steepest score-Jacobian regime) convergence behavior was never directly checked.

Also at 300 steps, cat↔dog's tangential_geodesic C values: resnet50=0.4999765, vit_base=0.4989030, clip=0.4631 — **more optimization did not push this pair decisively over 0.5**, it settled right at the boundary.

### 5.4 Config fixes from calibration

- **`geodesic_jvp_chunk_size` bug**: hardcoded at 8 (the 4GB-card value) regardless of profile — silently capped every `rtx5090` optimizer step at processing 8 segments regardless of the 5090's ~32GB. Fixed: profile-scoped, 64 for `rtx5090`.
- **Seeds cut 5→3**: not a rigor cut — SEED §3.3's own bar is "≥3 seeds"; 5 was unrequired padding.
- **`optimizer_steps` 500→150, `control_points` 32→16**: per §5.3.
- **Net effect**: full 45×3×3 sweep (real + permuted) estimated at **~42-51 hours**.

### 5.5 Resume/checkpoint cache — verified working

Built because the ~2-day sweep held everything in memory, writing to disk only at the very end. Tested on `local_smoke`: first run (0 cached, 6 computed) → second run (6/6 `[cached]`, seconds instead of ~2 min, byte-identical results and decision). Fingerprint-mismatch guard verified to correctly raise `RuntimeError` rather than silently mixing incompatible cached results.

### 5.6 `rtx5090` real sweep — in progress, σ=0.5 COMPLETE + σ=2.0 partial

At time of writing: 210/405 combos for `tangential_geodesic` (51.9%). **σ=0.5 is now fully complete — all 45/45 pairs, 3/3 seeds.** σ=2.0 has 25/45 pairs so far. Full per-pair tables: `results/gate/phase1_gate_full/posthoc_real_tangential_geodesic.json`.

**σ=0.5 (complete, all 3 evaluators):**

| Metric | Value |
|---|---|
| Strict routing events (C(A,B)>0.5, all evaluators+seeds) | **0 / 45** |
| Consistent argmax-flip events (weaker, relative bar) | **0 / 45** |
| avg `margin_runnerup` | 0.05–0.23 (low) |
| avg p(A or B) at the "other-class" peak moment | ~0.44–0.47 |
| avg worst-point-on-path p(A or B) | ~0.29–0.30 |

This portion of the sweep is now done, not partial — the null result on both bars is solid at this noise level.

**σ=2.0 (25/45 so far) — the theoretically-predicted "coarse structure should be more visible here" pattern is NOT showing up.** Directly comparing the same pairs across both σ levels (filtered resnet50+clip view):

| pair | σ=0.5 p(AB)@peak / min_p(AB) | σ=2.0 p(AB)@peak / min_p(AB) |
|---|---|---|
| airplane-automobile | 0.581 / 0.405 | 0.297 / 0.249 |
| airplane-cat | 0.503 / 0.246 | 0.336 / 0.174 |
| automobile-deer | 0.544 / 0.260 | 0.179 / 0.112 |
| airplane-frog | 0.450 / 0.260 | 0.186 / 0.110 (near the ~0.10 uniform floor) |

Confidence drops across the board at higher σ, but **`margin_runnerup` does not rise, and strict/flip are still 0/25 at σ=2.0.** This looks like "more noise → less confident about everything, uniformly" rather than "more noise → clearer coarse-class routing structure." The σ=2.0/8.0-will-show-it expectation from earlier in this report is **not being borne out so far** (σ=8.0 still untested).

**One genuinely interesting exception**: **automobile↔ship → consensus "airplane" at BOTH σ=0.5 and σ=2.0** — the only pair showing a reproducible, non-"disagree" consensus across two different noise levels tested so far. Still sub-threshold (never crosses strict), but intuitively coherent (automobile, ship, airplane are all vehicle-category classes) and the single most promising concrete data point in the sweep to date. Worth tracking as σ=8.0 and the remaining σ=2.0 pairs land.

**Filtered to resnet50+clip only** (dropping `vit_base`): pattern unchanged at both σ levels — still 0 strict/flip either way. The weak-evaluator hypothesis is a real but minor contributor, not the dominant explanation (a single earlier n=1 test had suggested otherwise — did not generalize).

**Realism read** (σ=0.5): `min_p(AB)≈0.29` sits meaningfully above the "fully unrecognizable" floor (~0.10-0.15 for 10-way uniform) but well below "stays confidently A-or-B" (~0.6+) — path midpoints land in a genuinely ambiguous zone, not distinguishable from softmax data alone between "realistic chimera our evaluators aren't trained to recognize" and "genuinely degraded/unrealistic image" (see §7). At σ=2.0 this ambiguity deepens further (values trending toward the uniform floor for several pairs).

### 5.7 Energy-scale sanity check (no bug found)

A raw `tangential_geodesic` energy value of ~44,000 at σ=0.5 (vs. ~540 in the σ=2.0 convergence check) was investigated and explained: energy is an unnormalized *sum* over all segments × batch, so it scales with `samples_per_class` (16 vs. 2, ~8x) and `num_control_points` (16 vs. 8, ~2x); the remaining gap is consistent with the score-Jacobian's magnitude growing sharply at low σ (`score = (denoised-x)/σ²`). Raw energy values are **not comparable across different σ/batch/control-point settings** — only relative decrease within one combo's own trajectory is meaningful.

### 5.8 Phase 0 Audit (`scripts/audit_cache.py`) — the most decisive result yet, and it's a clean null

Full findings in `PHASE0_AUDIT.md`; this is the `tangential_geodesic`-specific readout (254 combos, σ=0.5+σ=2.0 combined, n=762 individual C(A,B) values, properly stratified — see below for why stratification mattered):

**Check 2 (distribution shape)**: mean=0.2577, median=0.2505, std=0.0844. τ=0.5 sits **~2.9 standard deviations above the mean** — a smooth, unimodal, well-separated distribution, only 1.2% of values ever crossing τ. This is the cleanest evidence to date, and it cuts against the "maybe the threshold is just slightly too strict" reading: there is no suspicious clustering just below 0.5 that a lower τ would meaningfully unlock. **This looks like a genuine, well-powered null result for the strict routing claim**, independent of the permutation-control question (§5.9) — it's a direct property of the real sweep's own distribution.

**Check 3 (marginal argmax) — a real, substantial pattern, stronger than the pooled read suggested**: within `tangential_geodesic` specifically, **cat (23.1%) and dog (16.0%) together account for 39.1% of every "other-class" peak across 254 pairs** — roughly 2x what two uniformly-distributed classes would contribute (20% baseline). This is a genuine, path-specific effect (even more pronounced than in the pooled check-3 result from §5.6/audit v1), and it means **any pair-specific "routes through cat/dog" claim now needs real skepticism** — cat and dog look like generic attractor basins for this particular path-construction method, not evidence of pair-specific semantic structure. Airplane remains at 8.7% (below uniform), so the automobile↔ship→airplane finding (§5.6) is *not* explained by this effect — it survives as the one pair-specific result not attributable to a generic attractor class.

**Check 4 (evaluator ablation)**: clip=0.2405, resnet50=0.2762 (highest mean *and* highest std=0.1052), vit_base=0.2565 (diff from pooled others: **-0.0018**, i.e. essentially zero). The weak-evaluator hypothesis is now cleanly refuted a second time, at full power — and if anything **resnet50**, not vit_base, shows the most spread. Worth correcting the record on this.

**Check 5**: still untestable — permutation sweep for `all_pairs` mode hasn't started yet.

### 5.9 Permutation control — confirmed broken for both mechanisms (see `PHASE0_AUDIT.md` for the full code trace)

Two separate, confirmed problems, found by tracing source code directly (not speculation):

- **`linear_condition`**: a genuine masking bug. The conditional generator is the same checkpoint for real and permuted sweeps and never sees the permutation; images generated for slot `(a,b)` are byte-identical between the two sweeps. `compute_C` masks `{a,b}` (raw slot integers) instead of `{π(a),π(b)}`, so the permuted evaluator's *correct* recognition of the real content gets counted as fake routing — the actual mechanism behind the ~0.97-1.0 permuted values observed throughout this project for path 1, previously misattributed to "path 1 is an unreliable baseline."
- **`slerp_noise`/`tangential_geodesic`**: structural non-independence. Since every class has exactly 5000 images and `all_pairs` mode exhaustively covers all 45 label-pairs, the permuted sweep computes geodesics on the *same 45 underlying real-class pairs* as the real sweep, just relabeled — not an independent test of label-dependence.

**Given §5.8's clean, smooth, well-separated null distribution for `tangential_geodesic`, this matters less than it would have if the result had been positive** — a null result doesn't need a broken control to explain it away. And §5.10 now supplies a *working* null (A↔A same-class) that replaces the broken permutation control entirely. NB: an earlier version of this section proposed fixing the control via per-image random relabeling — the review correctly rejected that (random labels cripple the evaluator → ~10% accuracy → near-uniform softmax → the control passes mechanically for reasons unrelated to routing). The correct replacement keeps the evaluator real and is what §5.10 implements.

### 5.10 A↔A same-class null (`scripts/run_null_controls.py`) — the decisive comparison

C(A,B) is a double maximum (over ~16 control points × 8 classes), so it is upward-biased by selection alone and **uninterpretable in absolute terms without a null computed with the same machinery** (review §3.3). The A↔A control runs the identical geodesic pipeline between two *distinct samples of the same class* — where there is no third class to route to, so its C is the pure floor from midpoint ambiguity + double-max. Comparison against the real cross-class sweep, stratified by σ.

**Mask-matched result (review fix 1 applied — A↔A now masks its class + a 2nd random class → max over 8, identical to the real sweep; all 10 classes × 3 seeds, n=90 each side):**

| σ | real cross-class C mean | A↔A null C mean | gap (real − null) | reading |
|---|---|---|---|---|
| 0.5 | 0.2271 (n=405) | **0.1897** (n=90) | **+0.037** (~2.6σ) | real exceeds the within-class floor |
| 2.0 | 0.2932 (n=381) | **0.2926** (n=90) | **+0.0006** | identical — no cross-class-specific signal |

(The earlier estimate here was +0.068 at σ=0.5, from an underpowered mask-1 null of n=7 combos / 1 seed. Fixing the masking *and* the power deflated it to +0.037 — good hygiene: the fix made the number honest and the finding survived. The mask-matched numbers above supersede the earlier ones.)

Two decisive readings:

1. **The artifact floor is large and grows with σ** (0.190 → 0.293). Most of the "routing signal" in the raw C numbers is interpolation/selection artifact present even with no third class — the review's core concern, fully vindicated. Absolute C is nearly meaningless without this null.
2. **A genuine cross-class excess survives the null, but only at low σ.** At σ=0.5, real (0.2271) beats the matched floor (0.1897) by +0.037 (~2.6σ, real but modest). By σ=2.0 the gap is +0.0006 — gone; the floor rose to meet it. **Whatever weak cross-class structure exists lives at low noise and is erased by σ=2.0.** This inverts the earlier "σ=8.0 is the last hope" framing: the signal is at *low* σ, not high, so σ=8.0 (even blurrier) is the least likely place to find anything.

**Secondary finding — inter-evaluator agreement is not protective.** Evaluators agree on the peak class *more* on the A↔A null (σ=2.0: 50% all-agree) than on real cross-class pairs (σ=2.0: 21.3%); and on the σ=0.5 null their per-combo C-value *correlations* fall to near-zero (clip↔resnet50 Pearson +0.10, clip↔vit_base +0.02 — the "equal marginals, no agreement = noise" signature). They co-agree on *artifacts* (the cat attractor) as much as or more than on real structure — so SEED's "≥3 evaluators must agree" criterion provides little protection against artifact-driven false positives. (Caveat: the near-zero σ=0.5 correlations are partly range-restriction — null C has compressed variance — so the peak-class agreement rate is the more robust concordance measure here; both point the same way.)

**cat is a generic attractor even in the mask-matched within-class null** (20% of σ=0.5 peaks, 31% of σ=2.0 peaks) — confirming cat/dog are basins of the method/evaluators, not routing destinations.

**Fix 3 done — A↔A floors for all three paths (mask-matched, n=90 each), and the soft bar is now in serious KILL trouble.** The soft "geometry beats baselines" claim rested on the path ordering `tangential (0.29) > slerp (0.25) > linear (0.09)` from the small `local_poc` run. But the A↔A **null floors reproduce that exact ordering**:

| path | σ=0.5 A↔A floor | σ=2.0 A↔A floor |
|---|---|---|
| `linear_condition` | 0.081 (σ=final) | — |
| `slerp_noise` | 0.156 | 0.255 |
| `tangential_geodesic` | 0.190 | 0.293 |

Since the ordering holds in a null with *no routing*, it is explained by each method producing differently-blurry midpoints (linear generates a clean single class → tiny floor; slerp/tangential blend through noise → higher floors), **not** by the geometry path finding more real routing. The fair test is **floor-subtracted excess** (real − own null), now complete for all paths (real means from the σ-stratified real audit, floors from `phase1_null_same_class`):

| path | σ | real C | A↔A floor | **floor-subtracted excess** |
|---|---|---|---|---|
| `linear_condition` | final | 0.2295 | 0.0814 | **+0.148** |
| `slerp_noise` | 0.5 | 0.1991 | 0.1564 | **+0.043** |
| `slerp_noise` | 2.0 | 0.2621 | 0.2552 | +0.007 |
| `tangential_geodesic` | 0.5 | 0.2271 | 0.1897 | **+0.037** |
| `tangential_geodesic` | 2.0 | 0.2932 | 0.2926 | +0.001 |

**The soft "geometry beats baselines" bar is not merely unsupported — it is REVERSED.** After removing each method's own artifact floor, the geometry-aware tangential geodesic has the *smallest* low-σ cross-class excess: `slerp_noise` (a baseline) matches it (+0.043 vs +0.037, within noise), and `linear_condition` (the weakest baseline) has ~4× more (+0.148). So on the C metric there is **no routing advantage for the geometry path** — this is a **KILL for the routing hypothesis (RQ2/RQ3) as operationalized via C magnitude.**

**Crucial caveat — what C cannot see, and why a PIVOT is still open.** linear's large +0.148 is the *condition-blend chimera* effect (SEED §2's "Object 1": blending two class *conditionings* produces ambiguous mush that classifiers read as a third class). C cannot distinguish "a realistic member of a third class" from "a chimera that merely fools a classifier." So linear having the highest excess is consistent with it producing the *least realistic* thirds, exactly as the original hypothesis predicted — the metric just can't tell. **The only surviving route to a positive result is RQ4/realism**: does `tangential_geodesic` produce *realistic* third-class midpoints where `linear_condition` produces chimeras? That question is invisible to C and requires the realism analysis (visual inspection + a realism metric — FID/LPIPS/distance-to-manifold), which has **not** been done. This is now the decisive remaining measurement, ahead of everything else.

**Fix 2 (excursion-depth control) is now demoted.** It would clarify the *nature* of the shared ~0.04 low-σ excess, but since slerp shows the same excess as tangential, that excess is *not* a geometry-path-specific property — so it no longer bears on the soft bar (already answered: no advantage). Realism analysis supersedes it.

---

## 6. Known gaps / limitations (explicit, not yet addressed)

1. **No σ=8.0 data — and per §5.10 it is now LOW priority, not high.** The A↔A null shows the only cross-class signal lives at *low* σ (0.5) and is already gone by σ=2.0; σ=8.0 (blurrier still) is the least likely place to find routing, not the "last hope" earlier framing assumed. The live thread is σ≤0.5, not σ=8.0.
2. **Permutation control is confirmed broken for both mechanisms** (§5.9) — needs a redesign (per-image random relabeling) before any final KILL/GO claim can cite it, though this matters less now that §5.8 gives an independent, non-permutation-dependent reason to expect a null result.
3. **No direct realism metric** (FID, LPIPS, likelihood, distance-to-manifold) — everything so far is a classifier-softmax proxy. SEED §5 explicitly warns against collapsing these.
4. **No raw images saved anywhere in the cache** — only evaluator outputs. Can't visually inspect any generated path without a targeted re-run.
5. **`vit_base`'s 70% accuracy is unexplained** — underfit vs. something else not determined (no training curve reviewed).
6. **Evaluator peaks are found independently per-evaluator** (`compute_C` finds each evaluator's own best t) — "disagree" means each evaluator's *best individual evidence* points to different classes, not necessarily simultaneous disagreement at one fixed t.
7. **String-method path (4) never implemented** — deliberately deferred (SEED design), reserved for confirmatory checks on flagged pairs only, not yet relevant at this stage.

---

## 7. Current read — KILL on the C-based routing claim; RQ4/realism is the only surviving positive route

The A↔A null, now complete for all three paths (§5.10), settles both bars:

- **Strict routing bar: dead.** 0/45 confirmed events at σ=0.5 (complete), C distribution ~2.9σ below τ, no near-threshold clustering (§5.8). Robust and threshold-independent.
- **Absolute C magnitude: mostly artifact.** At σ=2.0 real cross-class C (0.2932) *equals* the within-class floor (0.2926) — zero routing-specific signal. Raw C is uninterpretable without this null; the review's core concern is fully vindicated.
- **Soft "geometry beats baselines" bar: REVERSED → KILL.** Floor-subtracted cross-class excess (§5.10 table): linear +0.148 ≫ slerp +0.043 ≈ tangential +0.037. The geometry-aware path shows the *least* cross-class activation, not the most. On the C metric the geometry path has no routing advantage over the naive baselines — the central RQ2/RQ3 claim is not supported and is if anything inverted.
- **The one door still open is RQ4/realism, and C is blind to it.** linear's high excess is the condition-blend *chimera* effect (SEED §2 Object 1) — C cannot tell a realistic third-class member from a classifier-fooling chimera. So "linear activates thirds most" is consistent with linear producing the *least realistic* thirds, exactly as hypothesized. Whether `tangential_geodesic` produces *realistic* midpoints where the baselines produce chimeras is the only remaining path to a positive result, and it requires a realism metric + visual inspection — **not yet measured.**
- **Two robust secondary findings** (n=90-135): evaluator agreement is *not* protective (evaluators co-agree on the cat artifact as much as/more than on real pairs); cat/dog (and deer/frog for slerp) are generic attractor basins, appearing top even in within-class interpolation.
- The weak ~0.04 low-σ cross-class excess is real but *shared* by slerp and tangential — a generic "crossing classes lights up thirds a bit at low noise" effect, not a geometry-path property. Gone by σ=2.0.

**Verdict: KILL on RQ2/RQ3 (routing, as measured by C) — the geometry path shows no routing advantage over baselines.** Per SEED §3.5, a **PIVOT to RQ4** (which realism objective produces the most on-manifold / least-chimeric transitions) remains available *only if* the realism analysis shows a tangential-vs-baseline realism gap. Absent that, this is a clean, well-controlled negative result. The realism measurement is now decisive and comes before all other pending items.

## 8. Recommended next steps, in priority order

1. **THE decisive measurement: realism, tangential vs. baselines.** This is the only thing that can turn the KILL into a PIVOT (RQ4). Two parts: (a) **visual inspection** (`scripts/inspect_path_images.py`, built) of matched midpoints — same pair, same σ, `tangential_geodesic` vs `slerp_noise` vs `linear_condition` — does the geometry path look like realistic third-class members where linear looks like mush/chimeras? (b) a **quantitative realism metric** (distance-to-manifold via the score norm is cheapest and already available from the model; LPIPS/FID if wanted) on the decoded midpoints, per path type. If tangential is measurably more realistic at equal C, that's the RQ4 paper; if not, it's a clean negative result.
2. **Do NOT resume σ=8.0** — least informative condition; the signal (such as it is) lives at low σ. Stop the sweep (resumable if ever reversed).
3. **Write-up framing follows from step 1.** Either: (a) RQ4 PIVOT — "different realism objectives induce different chimera rates; the manifold-tangential geodesic produces more realistic transitions than condition-interpolation at matched classifier-crossing" (if realism gap confirmed); or (b) characterized KILL — "curvature-weighted geodesics show no semantic-routing advantage over naive baselines on CIFAR-10; the apparent routing signal is a double-max interpolation artifact reproduced by a same-class null, and the geometry path if anything crosses classes *less* than condition-interpolation." Both are legitimate, honest science-of-generative-models results.
4. **Lower priority / optional**: sharpen the shared low-σ excess (real+null at σ<0.5) — but note it's not geometry-specific, so it's a secondary characterization, not a headline. Fix 2 excursion-depth control, fixed permutation control, `vit_base` training curve — none gate the verdict.
