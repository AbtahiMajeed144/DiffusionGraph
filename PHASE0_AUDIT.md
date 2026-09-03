# Phase 0 Audit

Five checks proposed against the Phase 1 sweep, all doable without GPU (code review + reading the existing cache). This is bookkeeping, not a gate — but two of the five checks turned up findings serious enough that **every prior "permutation control is clean" conclusion in `EXPERIMENT_REPORT.md` needs to be read with this attached.**

**Bottom line up front:** Check 1 passes cleanly. Checks 2-4 need to be run against the live 5090 cache (`scripts/audit_cache.py`, built and locally verified — see below). **Check 5 is the important one: confirmed via direct code tracing (not speculation) that the label-permutation control has two separate, serious problems** — a genuine masking bug for `linear_condition`, and a structural non-independence issue for `slerp_noise`/`tangential_geodesic` (the path the entire project's central claim rests on).

---

## Check 1 — What tensor reaches the evaluators?

**PASS.** Verified by reading `paths/tangential_geodesic.py` and `paths/slerp_noise.py`: every image handed to an evaluator comes from `EDMDenoiser.denoise_to_clean()`, which runs the **full reverse ODE from σ_tau down to 0** (a hand-rolled Heun integrator, `utils/edm_loader.py`) — not the raw noisy `x_t`, not a single-step Tweedie estimate. Evaluators only ever see fully-decoded, clean images. This gates the interpretability of every σ>0.5 number, and it checks out.

## Check 5 — Is the permutation control an identity? **Confirmed: two distinct real problems.**

Traced directly against `data/cifar10.py`, `paths/linear_condition.py`, `paths/tangential_geodesic.py`, `eval/routing.py`, and `scripts/run_gate.py`.

### 5a. `linear_condition` (path 1): a genuine masking bug

- `denoiser_cond` (the conditional generator) is the **same checkpoint** for both the real and permuted sweeps, trained with real label semantics it has no knowledge of the permutation for.
- `linear_condition.construct()` conditions on `onehot(class_a)`/`onehot(class_b)` — the raw integer slot values — and **never touches `x_a`/`x_b`'s pixel content at all** (only their `.shape`, for batch size).
- Consequence: for slot `(a,b)`, the permuted sweep generates a **byte-identical image** to the real sweep (same seed → same latents → same conditioning → same output).
- The permuted evaluator, shown this image, correctly recognizes the real content it actually is and reports that at *its own* index for that real class — which is `π(a)`, not `a`.
- `compute_C` (`eval/routing.py`) masks out `{trajectory.class_a, trajectory.class_b}` — **the raw slot integers**, not `{π(a), π(b)}`.
- **Result: the evaluator's correct classification isn't excluded, gets counted as "routing to a third class."** This is the exact, now-identified mechanism behind the ~0.97-1.0 permuted values observed for `linear_condition` throughout this project — previously (mis)read as "path 1 is an artifact-prone baseline," which was the wrong explanation. It was a masking bug the whole time.

### 5b. `slerp_noise` / `tangential_geodesic` (paths 2-3): structural non-independence, not a bug — but just as damaging to the control's validity

- Every CIFAR-10 class has exactly 5000 images. `CIFAR10Canonical.sample_pairs` draws via `torch.randperm(5000, generator=seed)` — the local index picks depend only on the count (always 5000) and the seed, **never on which class label they're filed under.**
- `class_pair_mode="all_pairs"` exhaustively iterates all C(10,2)=45 label-index pairs. Since the permutation π is a bijection over the 10 labels, this exhaustive coverage means: **the permuted sweep computes geodesics on the exact same 45 underlying real-class pairs as the real sweep** — just filed under shuffled slot names.
- These two path types use the **unconditional** checkpoint (`class_labels=None` always, confirmed in `tangential_geodesic.py` line 94) — the geometry computation never references labels at all, only the actual pixel content of `x_a`/`x_b`.
- Consequence: **for the same underlying real class pair, the generated path is identical (or statistically equivalent) between the real and permuted sweeps** — the only thing that differs is which comparably-accurate (but not identical) classifier scores it.
- This is not testing "does structure depend on genuine semantics." It is re-running the same 45 experiments under different bookkeeping, then asking two similarly-trained classifiers to independently grade them.
- **If real routing structure exists in the real sweep, it should appear equally in the permuted sweep, because they are the same 45 experiments.** A "permutation clean" result under this design provides **zero evidentiary protection** against the routing signal being a metric artifact — it was structurally guaranteed to look clean (or unclean) in lockstep with the real sweep, regardless of whether the underlying phenomenon is genuine.
- **Every "permutation clean, 0% false-positive rate" statement in `EXPERIMENT_REPORT.md` for `tangential_geodesic` needs to be read as "not yet meaningfully tested," not "confirmed clean."** This affects the central path type the whole project hinges on.

**Auditor's proposed fix, and it's the right one**: replace the bijective 10-label permutation (which preserves intact, coherent real classes just under new names) with a genuine **pair-shuffled / per-image-random-label null** — assign each image an independent, uniformly random label among 10, so "permuted class 3" is a genuinely incoherent mixture of all 10 real classes, not a renamed copy of one real class. That breaks the "same 45 real experiments" equivalence entirely, and is what the control was supposed to be testing in the first place. **Not yet implemented** — this needs a decision on whether to fix now (re-running just the permutation half, since real-sweep results are unaffected) or after the real-sweep σ=8.0 data lands.

## Checks 2-4 — run against the live cache, no GPU needed

Built and **locally verified end-to-end** as `scripts/audit_cache.py` (tested against a regenerated `local_smoke` cache — real code paths exercised, no placeholder logic). Run on the 5090:

```bash
git pull
python scripts/audit_cache.py --run-name phase1_gate_full
```

- **Check 2** (histogram of all C(A,B) values): reports the full distribution shape, not just a threshold count — flags explicitly if mass sits unimodally just below τ=0.5 (the honest "no separation from null" read) vs. a genuinely separated/bimodal distribution.
- **Check 3** (marginal argmax distribution across all pairs): flags any class appearing as the "other-class" peak disproportionately often globally — directly tests whether the automobile↔ship→airplane finding (`EXPERIMENT_REPORT.md` §5.6) is pair-specific structure or just "airplane" being the model's generic default guess for any ambiguous image.
- **Check 4** (`vit_base` ablation on continuous C, not event counts): reports each evaluator's own C(A,B) distribution (mean/median/std/percentiles) directly — the prior comparison (`analyze_cache.py --evaluators`) compared 0-vs-0 event counts, which is uninformative when nothing has crossed τ anyway; this compares the actual continuous values.
- **Bonus empirical check for 5b**: directly measures the correlation between real-pair(a,b)'s C values and permuted-pair(π(a),π(b))'s C values for the *same underlying real class pair* — a positive correlation would be direct empirical confirmation of the code-trace finding above, not just a theoretical argument.

## Checks 2-4, actual results (`--path tangential_geodesic`, 254 combos, σ=0.5+σ=2.0, n=762)

**Note on methodology**: the first run of this script pooled all path types together, which diluted exactly what these checks exist to answer (`linear_condition` runs much lower than `tangential_geodesic`). Fixed to stratify by path type by default; numbers below are `tangential_geodesic` alone.

- **Check 2**: mean=0.2577, median=0.2505, std=0.0844. τ=0.5 sits **~2.9σ above the mean**; only 1.2% of values ever cross it. Smooth, unimodal, well-separated — not a "clustered just below threshold" pattern that a slightly lower τ would unlock. **This is the strongest evidence in the project so far for a genuine null**, and it doesn't depend on the (confirmed-broken, above) permutation control at all.
- **Check 3**: cat (23.1%) + dog (16.0%) = **39.1%** of every "other-class" peak across 254 pairs — ~2x the 20% two-uniform-classes baseline. A real, path-specific attractor effect (worse than the pooled run suggested), meaning any pair-specific "routes through cat/dog" claim needs real skepticism now. Airplane stays at 8.7% (below uniform) — the automobile↔ship→airplane finding survives as the one result not explained by this effect.
- **Check 4**: clip=0.2405, resnet50=0.2762 (highest mean *and* std), vit_base=0.2565 (diff from others: **-0.0018**). Cleanly refutes the weak-evaluator hypothesis a second time, at full statistical power — if anything resnet50 is the noisiest evaluator, not vit_base.
- **Check 5**: still 0 permuted combos loaded — the `all_pairs`-mode permutation sweep hasn't started yet.

Full write-up in `EXPERIMENT_REPORT.md` §5.8-5.9.

## Deliverable status

This file is the "one-page honest restatement" the audit calls for. **Checks 1, 5, and now 2-4 (`tangential_geodesic`) are done.** Remaining: re-run checks 2-4 for `slerp_noise`/`linear_condition` for comparison (`--path` flag), and check 5's empirical correlation test once permuted `all_pairs` data starts landing. No gate blocks on this — it's bookkeeping needed regardless of which way Phase 1 ultimately reads, and the permutation control fix is still needed before any final GO/PIVOT/KILL call can cite it as evidence, even though the Check 2 result now stands on its own independent of that fix.
