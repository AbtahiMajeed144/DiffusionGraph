# Phase 1 Gate — Experiment Report

**Status as of this report: Phase 1 gate sweep in progress on the RTX 5090 (`rtx5090` profile). σ=0.5 and σ=2.0 combined (254 `tangential_geodesic` combos, n=762 individual values) now show a clean, well-powered, statistically-separated null distribution (Phase 0 Audit, §5.8) — the strongest evidence yet, and it doesn't depend on the permutation control, which is separately confirmed broken (§5.9) and needs fixing regardless. No final GO/PIVOT/KILL decision yet; the read is now meaningfully closer to PIVOT than at any earlier checkpoint, pending σ=8.0.** This document is the running record of every experiment, calibration, and finding behind that in-progress run.

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

**Given §5.8's clean, smooth, well-separated null distribution for `tangential_geodesic`, this matters less than it would have if the result had been positive** — a null result doesn't need a broken control to explain it away, and the distribution shape (no suspicious near-threshold clustering) argues against a metric-artifact story too. But it still needs fixing (per-image random relabeling, not a bijective class permutation) before any final KILL/GO claim can cite "permutation clean" as evidence — not yet done.

---

## 6. Known gaps / limitations (explicit, not yet addressed)

1. **No σ=8.0 data yet.** σ=0.5 is complete (null); σ=2.0/0.5 combined (254 `tangential_geodesic` combos) now show a clean, well-separated null distribution (§5.8) — the single biggest open question is whether σ=8.0 changes this, but the trend across two noise levels and a properly-powered distributional check is not encouraging for that hope.
2. **Permutation control is confirmed broken for both mechanisms** (§5.9) — needs a redesign (per-image random relabeling) before any final KILL/GO claim can cite it, though this matters less now that §5.8 gives an independent, non-permutation-dependent reason to expect a null result.
3. **No direct realism metric** (FID, LPIPS, likelihood, distance-to-manifold) — everything so far is a classifier-softmax proxy. SEED §5 explicitly warns against collapsing these.
4. **No raw images saved anywhere in the cache** — only evaluator outputs. Can't visually inspect any generated path without a targeted re-run.
5. **`vit_base`'s 70% accuracy is unexplained** — underfit vs. something else not determined (no training curve reviewed).
6. **Evaluator peaks are found independently per-evaluator** (`compute_C` finds each evaluator's own best t) — "disagree" means each evaluator's *best individual evidence* points to different classes, not necessarily simultaneous disagreement at one fixed t.
7. **String-method path (4) never implemented** — deliberately deferred (SEED design), reserved for confirmatory checks on flagged pairs only, not yet relevant at this stage.

---

## 7. Current read (not a final decision, but the most confident this report has been)

- **The strict bar (§5.8, Check 2) is now the strongest single piece of evidence in this project**: a smooth, unimodal, well-powered (n=762) distribution of `tangential_geodesic` C(A,B) values centered ~2.9 standard deviations below τ=0.5, with only 1.2% ever crossing. This isn't "0 events because the threshold is slightly too strict" — the shape itself shows no separation from a null distribution. This reading does not depend on the (confirmed-broken, §5.9) permutation control at all, which makes it more trustworthy than anything that came before it, not less.
- **The soft bar** (geometry-aware path beats baselines on average) held in the small-scale `local_poc` run (0.293 vs 0.247/0.091) but has not been recomputed at the real sweep's scale — worth doing, but Check 2's distributional evidence is a stronger signal than a single mean comparison either way.
- **The theoretical expectation that higher σ would reveal more routing structure is still not supported** across σ=0.5 and σ=2.0 combined — confidence drops with noise rather than concentrating onto specific third classes. σ=8.0 remains the last untested condition, but two levels of consistent null result lower the odds it reverses the picture.
- **cat/dog as generic attractor classes (39.1% of all "other-class" peaks, §5.8 Check 3) is itself a real, notable finding** — worth understanding on its own terms (a property of the tangential-geodesic method's geometry, or of the evaluators' training, not yet determined) — and it means the one previously-promising lead, automobile↔ship→airplane, is the *only* pair-specific result not explainable by this generic-attractor effect.
- The realism data (§5.6) still suggests the geodesic path isn't achieving clean, confident samples throughout — a live, separate question from whether it routes.
- **Not yet a final decision** — σ=8.0 is untested, the permutation control needs fixing before being cited as evidence either way, and no visual inspection has happened yet. But this is meaningfully closer to PIVOT than any previous checkpoint in this report, on stronger evidence than before.

## 8. Recommended next steps, in priority order

1. **Let σ=2.0 finish and σ=8.0 land, re-run `analyze_cache.py`** — σ=2.0's partial trend doesn't look promising, but σ=8.0 (untested) is the last chance for the noise-level hypothesis to hold. If it shows the same uniform-confidence-decay pattern, that's a strong signal toward PIVOT.
2. **Run the permutation-control comparison** once enough real data exists, to check the σ=0.5 null isn't secretly an artifact story instead.
3. **Visually inspect actual images** for 2-3 contrasting pairs (highest/lowest `min_p(AB)`) — the only way to resolve the realistic-chimera-vs-genuinely-broken question in §6.3. Requires a small targeted re-run with image-saving added (not yet built).
4. **Investigate `vit_base`'s training curve** — rule in/out underfitting before trusting or discounting its votes further.
5. Once σ=2.0/8.0 + permutation data are in: recompute the full GO/PIVOT/KILL decision on both bars, and decide whether Phase 1's "specific identifiable intermediate class" framing (the strict bar) was the right operationalization of SEED's central claim, or whether the softer magnitude-contrast + realism-focused reading is the more defensible one to report.
