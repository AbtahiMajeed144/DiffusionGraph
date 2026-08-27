# Phase 1 Gate — Experiment Report

**Status as of this report: Phase 1 gate sweep in progress on the RTX 5090 (`rtx5090` profile), ~26% complete, σ=0.5 only so far. No final GO/PIVOT/KILL decision yet.** This document is the running record of every experiment, calibration, and finding behind that in-progress run — written to support the σ=2.0/σ=8.0 decision and the eventual Phase 2 go/no-go call, not as a final result.

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

### 5.6 `rtx5090` real sweep — in progress, σ=0.5 partial data

At time of writing: **106-107/405 combos for `tangential_geodesic`, all σ=0.5 (26.2% coverage)**, 35 pairs with full 3-seed coverage. Full per-pair table: `results/gate/phase1_gate_full/posthoc_real_tangential_geodesic.json`.

**Aggregate, all 3 evaluators:**

| Metric | Value |
|---|---|
| Strict routing events (C(A,B)>0.5, all evaluators+seeds) | **0 / 35** |
| Consistent argmax-flip events (weaker, relative bar) | **0 / 35** |
| Consensus intermediate class | "disagree" for ~31/35 pairs |
| avg `margin_runnerup` | 0.05–0.21 (low — no class clearly dominant even among the 8 non-endpoint classes) |
| avg p(A or B) at the "other-class" peak moment | 0.4286 |
| avg worst-point-on-path p(A or B) | 0.2874 |

**Filtered to resnet50+clip only** (dropping `vit_base`): still **0/35 strict, 0/35 flip** — only 4/35 pairs gain a clean single-class consensus. The weak-evaluator hypothesis is a real but minor contributor, not the dominant explanation (a single earlier n=1 test had suggested otherwise — did not generalize).

**Realism read**: `min_p(AB)≈0.29` sits meaningfully above the "fully unrecognizable" floor (~0.10-0.15 for 10-way uniform) but well below "stays confidently A-or-B" (~0.6+). Combined with the low margins and pervasive evaluator disagreement, the most honest characterization is: **path midpoints are landing in a genuinely ambiguous zone — probability spread across several classes rather than concentrated on the endpoints, a specific third class, or nothing.** This is not yet distinguishable, from softmax data alone, between "realistic chimera our evaluators aren't trained to recognize" and "genuinely degraded/unrealistic image" — see §7.

**Important caveat on all of §5.6: σ=0.5 is the *lowest*-noise level tested, and both Park et al. and `Strategic_Blind_Spots_Analysis.md` (#1) predict coarse, shape-level routing structure should appear at *higher* noise, not low — low-σ is close to the finished image, where fine detail (not class identity) is what's still being decided. A null result at σ=0.5 alone does not predict what σ=2.0 or σ=8.0 will show.**

### 5.7 Energy-scale sanity check (no bug found)

A raw `tangential_geodesic` energy value of ~44,000 at σ=0.5 (vs. ~540 in the σ=2.0 convergence check) was investigated and explained: energy is an unnormalized *sum* over all segments × batch, so it scales with `samples_per_class` (16 vs. 2, ~8x) and `num_control_points` (16 vs. 8, ~2x); the remaining gap is consistent with the score-Jacobian's magnitude growing sharply at low σ (`score = (denoised-x)/σ²`). Raw energy values are **not comparable across different σ/batch/control-point settings** — only relative decrease within one combo's own trajectory is meaningful.

---

## 6. Known gaps / limitations (explicit, not yet addressed)

1. **No σ=2.0 or σ=8.0 data yet** — the single biggest open question; see the caveat in §5.6.
2. **No permutation-control comparison for the real `rtx5090` sweep** — still pending, needed for the KILL check.
3. **No direct realism metric** (FID, LPIPS, likelihood, distance-to-manifold) — everything so far is a classifier-softmax proxy. SEED §5 explicitly warns against collapsing these.
4. **No raw images saved anywhere in the cache** — only evaluator outputs. Can't visually inspect any generated path without a targeted re-run.
5. **`vit_base`'s 70% accuracy is unexplained** — underfit vs. something else not determined (no training curve reviewed).
6. **Evaluator peaks are found independently per-evaluator** (`compute_C` finds each evaluator's own best t) — "disagree" means each evaluator's *best individual evidence* points to different classes, not necessarily simultaneous disagreement at one fixed t.
7. **String-method path (4) never implemented** — deliberately deferred (SEED design), reserved for confirmatory checks on flagged pairs only, not yet relevant at this stage.

---

## 7. Current read (not a final decision)

- The **soft bar** (geometry-aware path beats baselines on average) held clearly in every complete small-scale run so far (`local_poc`: 0.293 vs 0.247/0.091). Not yet recomputed at the partial `rtx5090` scale.
- The **strict bar** (confirmed individual routing events) has now been tested on 35 real pairs at σ=0.5 and found **zero** — a materially weaker result than the small-n checks suggested.
- The realism data suggests the geodesic path is not achieving clean, confident samples throughout — a genuine open question about whether `tangential_geodesic` is meeting its own design goal (manifold-tangential ⇒ realistic), separate from whether it routes.
- **None of this is conclusive yet.** σ=0.5 alone, no permutation comparison, no visual inspection, no higher-σ data — this is a checkpoint, not a verdict.

## 8. Recommended next steps, in priority order

1. **Let σ=2.0 and σ=8.0 data land, re-run `analyze_cache.py`** — the theoretically-motivated place to actually see routing, if it exists.
2. **Run the permutation-control comparison** once enough real data exists, to check the σ=0.5 null isn't secretly an artifact story instead.
3. **Visually inspect actual images** for 2-3 contrasting pairs (highest/lowest `min_p(AB)`) — the only way to resolve the realistic-chimera-vs-genuinely-broken question in §6.3. Requires a small targeted re-run with image-saving added (not yet built).
4. **Investigate `vit_base`'s training curve** — rule in/out underfitting before trusting or discounting its votes further.
5. Once σ=2.0/8.0 + permutation data are in: recompute the full GO/PIVOT/KILL decision on both bars, and decide whether Phase 1's "specific identifiable intermediate class" framing (the strict bar) was the right operationalization of SEED's central claim, or whether the softer magnitude-contrast + realism-focused reading is the more defensible one to report.
