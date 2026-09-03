# Phase 1 Gate — Experiment Report

**Status as of this report: read is PIVOT, on strong evidence (§5.10).** The A↔A same-class null (mask-matched, n=90/σ) resolves the picture: at σ=2.0 real cross-class C (0.2932) equals the within-class artifact floor (0.2926) → the geodesic instrument's routing signal is dominated by an interpolation/selection artifact, not semantics; at σ=0.5 a small genuine cross-class excess survives the matched null (+0.037, ~2.6σ), the only real structure found, and it lives at *low* noise (so σ=8.0 is now low-priority, not "the last hope"). Strict routing bar is a clean, threshold-robust null. Recommendation: stop the sweep, do not resume σ=8.0, pull the one live thread (low-σ excess) then write up as a characterized PIVOT. This does not depend on the (separately confirmed-broken, §5.9) permutation control. This document is the running record of every experiment, calibration, and finding.

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

Since the ordering holds in a null with *no routing*, it is explained by each method producing differently-blurry midpoints (linear generates a clean single class → tiny floor; slerp/tangential blend through noise → higher floors), **not** by the geometry path finding more real routing. The soft-bar ordering is largely a floor artifact. The fair test is **floor-subtracted excess** (real − own null). For `tangential_geodesic`: +0.037 (σ=0.5), +0.001 (σ=2.0). To finalize KILL-vs-PIVOT, the same subtraction is needed for slerp/linear — pending the real sweep's slerp/linear C means (they ran first in the sweep, so they are cached):
`python scripts/audit_cache.py --run-name phase1_gate_full --path slerp_noise` and `--path linear_condition`. If slerp's low-σ floor-subtracted excess ≈ tangential's +0.037, the geometry path has no real edge → the finding is a KILL on the soft bar with only the single, method-agnostic low-σ cross-class excess surviving.

**Still pending (review fix 2):** the excursion-depth control — is the +0.037 genuine routing or just that cross-class endpoints are farther apart (longer geodesic → more ambiguous midpoints)? The review's "random-endpoint" / `decoupled` null vs. a cleaner far-apart-same-class variant. Construction choice open.

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

## 7. Current read — PIVOT, on the strongest evidence in the project

The A↔A null (§5.10) turns the earlier "clean null" into a precise, quantitative account:

- **Strict routing bar: dead.** 0/45 confirmed events at σ=0.5 (complete), and the C distribution sits ~2.9σ below τ with no near-threshold clustering (§5.8). Robust and threshold-independent.
- **Absolute C magnitude: mostly artifact.** The A↔A null measures the double-max floor directly (0.159 at σ=0.5, 0.293 at σ=2.0). At σ=2.0 real cross-class C (0.2932) *equals* the within-class floor (0.2930) — zero routing-specific signal. The review's central concern is fully vindicated: raw C is uninterpretable without this null.
- **One genuine, narrow signal.** At σ=0.5, real cross-class C (0.2271) exceeds the mask-matched A↔A floor (0.1897) by +0.037 (~2.6σ, n=90 null). This is real — geodesics between *different* classes put more mass on third classes than within-class geodesics — but small, sub-threshold, and confined to low noise (gone by σ=2.0). It is not "routing events"; it is a weak distributional excess, and still needs the excursion-depth control (§5.10 pending) to confirm it's about class-crossing rather than geodesic length.
- **Evaluator agreement is not protective** (§5.10) — evaluators co-agree on the A↔A artifact as much as or more than on real pairs. The ≥3-evaluator criterion doesn't guard against artifact false positives.
- **cat/dog are generic attractor basins**, appearing as the top "other class" even in within-class interpolation — not evidence of pair-specific routing. automobile↔ship→airplane remains the one pair-specific observation not explained by a generic attractor, still n=1-ish.

**Verdict: PIVOT** per SEED §3.5 — no confirmed routing, the geodesic instrument's signal is dominated by an interpolation/selection artifact, but there is a real (if weak, low-σ-only) cross-class structure worth one focused look before the instrument is fully retired. This does *not* require σ=8.0 or a fixed permutation control to conclude.

## 8. Recommended next steps, in priority order

1. **Do NOT resume σ=8.0.** The signal lives at low σ; σ=8.0 is the least informative condition. Stop the sweep (it's resumable if this call is ever reversed).
2. **Firm up and sharpen the low-σ result** — the one live thread: (a) extend the A↔A null to all 10 classes × 3 seeds and fix the masking asymmetry (mask a 2nd random class → max over 8, matching real) for an airtight quantitative gap; (b) run the null and real at σ *below* 0.5 (e.g. 0.1, 0.25) to see whether the cross-class excess grows as class structure sharpens.
3. **Visually inspect images** (`scripts/inspect_path_images.py`, already built) for the highest-gap σ=0.5 pairs vs. an A↔A example — resolves the realistic-chimera-vs-degraded question and shows what, if anything, the low-σ excess corresponds to visually.
4. **Decide Phase 1 framing**: write it up as a characterized PIVOT — "the curvature-weighted geodesic instrument measures an interpolation artifact, not semantic routing, on CIFAR-10; a weak low-noise cross-class excess is the only structure that survives a same-class null" — which is a legitimate, honest science-of-generative-models negative result with one positive thread, not a failure.
5. Deferred/lower value now: the decoupled control (confirm masking semantics first), the fixed permutation control, `vit_base` training curve. None gate the PIVOT call.
