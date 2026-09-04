# Graph-Based Realism Barrier Experiment — CIFAR-10

## 0. What this measures, and what it does not

**Object.** For a realism score `R(x)` (higher = more realistic), the barrier between
classes A and B is

```
    τ*(A,B)  =  max        min  R( γ(t) )
              γ: A → B      t
```

the highest realism floor that can be maintained on some A→B path. Equivalently, in
superlevel-set form: with `S_τ = {x : R(x) ≥ τ}`,

```
    τ*(A,B) = sup { τ : A and B lie in the same connected component of S_τ }
```

The graph experiment is a *discrete estimator* of this quantity.

**What it does not measure.** Nothing here is a geodesic, a path length, or a
classifier statistic. It shares no machinery with the Phase-1 sweep except the
denoiser and the CIFAR data. A null in Phase 1 says nothing about this.

**The two opposing biases, stated up front.** They must both be reported, because
neither can be removed:

| Bias | Direction | Diagnostic |
|---|---|---|
| **Coverage**: the graph can only route through nodes it has. A better path may exist that we never sampled. | τ* **under**-estimated (barrier looks worse than it is) | Node-count ablation (§7.1) |
| **Resolution**: an edge's weight is a finite-sample min along the segment; it misses dips between samples. | τ* **over**-estimated (barrier looks better than it is) | Segment-resolution ablation (§7.2) |

A τ* that is stable under both is trustworthy. If it moves a lot under either, report
the sensitivity curve, not a point estimate.

---

## 0.4 Where this sits, and what it reuses

**Relationship to Phase 1.** Phase 1 measured a *classifier* statistic `C` along
geometry-aware paths and returned a clean, well-controlled negative: after subtracting
a same-class null, the tangential-geodesic path shows **no** cross-class "routing"
advantage over naive baselines (if anything it activates third classes *less* than
condition-interpolation). `C` is blind to realism — it cannot separate a realistic
third-class member from a chimera that merely fools a classifier. This experiment
swaps the classifier statistic for a *realism* statistic `R` and asks a different,
**path-free** question: are two classes joined by a route that stays realistic
throughout? `τ*` is therefore the RQ4/realism object Phase 1's result pointed to,
computed with entirely separate machinery. **The independence is two-way:** a Phase-1
`C`-null does not constrain `τ*`, and a `τ*` result does not resurrect the Phase-1
routing claim. Keep the two out of the same sentence in any write-up.

**Reused vs. new — executable grounding.** Most of the denoiser-facing work already
exists in the repo; the graph machinery is new but pure-CPU.

| Component | Reuse (existing module) | New |
|---|---|---|
| Denoiser, score, decode-to-clean | `utils/edm_loader.py` — `EDMDenoiser.score`, `.denoise_to_clean`; **unconditional** checkpoint (paths 2–4 convention) | — |
| SCOPED score (Stage 0/2) | `utils/jvp.py` — `score_jvp` **and** `hutchinson_trace` (already written for the geodesic; SCOPED = score-norm² + Jacobian-trace via exactly these) | thin wrapper only |
| Anchors & class membership | `data/cifar10.py` — `CIFAR10Canonical(train=False)` **test** split (EDM never saw it) | — |
| slerp interpolant nodes | `paths/slerp_noise.py` (σ=0.5, `denoise_to_clean`) | — |
| pixel-blend interpolant nodes | — | small: linear blend in `[-1,1]` pixel space, no decode |
| `G_confusion`, `G_CLIP` (Stage 7) | `models/classifiers.resnet50`, `models/embeddings.ClipZeroShot` | compute confusion on test set (trivial) |
| `G_fwdback` `T_σ` (Stage 7, the falsifier) | `EDMDenoiser` forward-diffuse + `denoise_to_clean` + resnet50 label | small |
| graph / k-NN / union-find / barriers / excision / controls / sensitivity | — | all new, numpy/scipy, no GPU |

Reuse of `jvp.py` for SCOPED is the single biggest cost saver — it collapses Stage-0
candidate #2 and the whole 17k-node scoring to "minutes" (see budget), and it is code
that has already been unit-exercised this project.

---

## Stage 0 — Validate the realism score (HARD GATE)

Do not build a graph before this passes. Everything downstream is a monotone function
of R, so a wrongly-ordered R produces a confidently wrong tree.

### 0.1 Four groups, 500 images each

| Group | Contents | Expected R |
|---|---|---|
| **G1** real | CIFAR-10 *test* images (not train — the EDM saw train) | highest |
| **G2** synthetic | unconditional EDM samples, full reverse ODE from σ_max | ≈ G1 |
| **G3** interpolant midpoints | slerp-in-noise and tangential-geodesic midpoints (t=0.5), decoded — reuse Phase-1 pairs | lower than G1/G2 |
| **G4** degraded | three subgroups of ~167: (a) **pixel-space linear blends** of two class images at t=0.5 (double-exposure ghosting), (b) Gaussian blur σ_blur ∈ {1,2,3}px, (c) additive Gaussian noise at SNR matched to (b)'s perceptual damage | lowest |

### 0.2 Candidate scores

Run all five. Include the last as a **negative control** — it is predicted to fail.

| Candidate | Cost/eval | Why included |
|---|---|---|
| **Graham reconstruction** (arXiv:2211.07740, CVPR 2023) | ~637 NFE (PLMS, 13 reconstructions) | most direct "on the model's manifold"; affordable under this design |
| **SCOPED** (arXiv:2510.01456) | 1 forward + 1 JVP | fastest; **directly reuses `utils/jvp.py`** (`score_jvp` + `hutchinson_trace`). Test statistic = score-norm² + Jacobian-trace, KDE-thresholded |
| **EigenScore** (arXiv:2510.07206, ICLR 2026) | moderate; Jacobian-free subspace iteration (forward-only) | built specifically for the near-OOD (CIFAR-10 vs -100) inversion mode that sinks naive scores |
| **Feature kNN** (Sun et al. 2022), CLIP or resnet50 penultimate | 1 forward | non-diffusion control; if this wins, the result is not diffusion-specific |
| **Raw score norm** ‖∇log p_σ‖ = ‖D(x;σ)−x‖/σ² | 1 forward | **negative control.** Vanishes at modes *and* saddles → predicted to fail G3-vs-G4b |

(Citations for the three diffusion candidates verified 2026-09-04. SCOPED and EigenScore
both explicitly target the near-OOD regime this experiment lives in — the strongest
prior reason to expect at least one to clear the decisive §0.3 criterion 2.)

### 0.3 Pass criteria

Report per candidate:

1. **AUROC(G1∪G2 vs G4)** — the easy far-ish separation. Need ≥ 0.90 to proceed.
2. **AUROC(G3 vs G4a)** — **the decisive test.** Both are class-mixed midpoints; one is
   manifold-aware, one is a literal double exposure. A score that cannot separate
   these is not measuring realism, it is measuring "is this image mixed." Need ≥ 0.75.
3. **Median ordering** G1 ≈ G2 > G3 > G4, with no inversions.
4. **Monotone response to graded damage**: R must decrease monotonically across
   σ_blur ∈ {0,1,2,3}. A non-monotone response invalidates the score for barrier use,
   because the barrier lives on the graded slope, not at the extremes.

5. **Off-midpoint stability (distribution match).** Stage 2 evaluates `R` on pixel
   blends `(1−s)u + s·v` at *all* `s ∈ (0,1)`, not just `s=0.5` — G4a only covers the
   `s=0.5` worst case. Spot-check `R` on blends at `s ∈ {0.25, 0.75}` for ~200 pairs
   and confirm it degrades *smoothly* from endpoint to midpoint. A score sharp at
   `s=0.5` but erratic off-midpoint makes edge weights (a min over interior `s`) noisy
   and the whole graph unstable. This is a weaker gate than criteria 1–4 (no numeric
   threshold), but a visibly non-smooth `R(s)` profile disqualifies a candidate that
   otherwise passes.

**Gate:** pick the cheapest candidate that passes criteria 1–4 (and is smooth under 5).
If none passes criterion 2, **stop** — the whole program is blocked on realism
estimation and that becomes the research problem.

**Deliverable:** one table, five rows, four columns, plus four score-distribution
histograms. Half a day.

---

## Stage 1 — Node set (this is where my earlier suggestion was wrong)

**Do not sample from `p_σ`.** Sampling a density to find its barriers is
self-defeating: samples concentrate in basins, the between-region is by definition
low-density, and the graph then contains an edge jumping straight from a cat node to
a dog node with nothing between — reporting a high barrier because it never sampled
the barrier. Populate the between-region *by construction* instead.

### 1.1 Composition

| Type | Count | Purpose |
|---|---|---|
| **Class anchors** | 10 × 32 = 320 real test images | path endpoints; also define class membership |
| **Manifold filler** | 2,000 unconditional EDM samples | alternative routes not on any interpolant; hub candidates |
| **Cross-class interpolants** | 45 pairs × 8 endpoint-pairs × 17 t-values × 2 path types = 12,240 | the between-region. Path types: slerp-in-noise (σ=0.5), pixel-space linear |
| **Same-class interpolants** | 10 × 8 × 17 × 2 = 2,720 | the within-class null (§6.1) |
| **Total** | **≈ 17,280** | |

Notes on the choices:

- **Include pixel-space linear interpolants deliberately.** They are ghosting
  double-exposures — genuinely low-R, and they sit exactly where the barrier is. They
  tell the graph where the wall is. Excluding them biases τ* upward.
- **Reuse Phase-1 geodesic control points** if they are still on disk; they are free
  extra coverage. If images were never saved (report §6.4), regenerate slerp only —
  don't re-run the optimizer for this.
- **All nodes stored as clean decoded images** in `[-1,1]^{3×32×32}`. Interpolants are
  constructed at σ_tau then decoded through the full reverse ODE, matching Phase 1's
  `denoise_to_clean`.
- **Provenance tag per node**: `(type, class_a, class_b, t, path_type, seed)`. Needed
  for §5 and §6.3. Store it; it costs nothing and you will want it.

### 1.2 Coverage limitation, honestly

The node set is biased toward *straight-ish* routes between class pairs. A genuinely
curved detour may be unrepresented. Two mitigations: multiple path types, and 8
distinct endpoint pairs per class pair (so a route can hop between *different* pairs'
interpolants — a cat→dog route may borrow a segment from a cat→deer path). This
cross-pair hopping is where globality comes from.

Residual risk is one-directional: **the graph can only find routes it has nodes for,
so τ* is a lower bound on the true barrier realism.** A *good* barrier found on the
graph is therefore real; a bad one may be an artifact of coverage.

---

## Stage 2 — Graph and edge weights

### 2.1 Locality

Compute pairwise L2 in **pixel space** (not feature space): the path we are implicitly
traversing is a pixel-space segment, so pixel distance is the quantity that controls
whether `min(R(u),R(v))` is a valid stand-in for the segment. 17k × 17k chunked on GPU
is seconds.

Build a symmetric k-NN graph, **k = 20**, plus a minimum-spanning-tree union to
guarantee connectivity. No hard ε threshold — see 2.2 for why ε is not needed as a
free parameter.

**Sanity-check locality before Stage 3** (cheap, one-time): pixel L2 is the right
metric *by construction* — §2.2's edge model traverses a literal pixel segment, so the
locality metric must be the segment metric — but it is only *useful* if the node set is
dense enough that a node's nearest neighbours are near-duplicates (short edges then
stay high-R, and only genuinely long edges pay the realism penalty). For ~20 random
nodes, confirm their pixel-L2 nearest neighbours are perceptually near-identical. If
instead nearest neighbours are visibly *different* images, the between-region is
undersampled — a Stage-1 coverage problem (§6.1), diagnose it there — not a reason to
switch to a feature metric, which would break the metric/edge-model coupling.

### 2.2 Edge weight — the key definition

Naive `w(u,v) = min(R(u), R(v))` is wrong: it ignores what happens *between* two nodes,
which is exactly the failure that made density-sampling useless. Instead, sample the
segment:

```
    w(u,v) = min over s ∈ {0, 1/S, 2/S, ..., 1}  of  R( (1−s)·u + s·v )

    with  S(u,v) = max(2, ceil( ‖u − v‖ / δ ))
```

where δ is a fixed step length (calibrate δ so that S=2 for a typical nearest-neighbour
distance). Long edges automatically get more interior samples and so are automatically
penalised for crossing bad territory. **This is what removes ε as a tuning knob.**

### 2.3 Making it affordable — lazy refinement

Evaluating S interior points on all ~170k edges is 10× the node budget. Don't. Iterate:

1. Initialise all edges with the optimistic `w = min(R(u), R(v))` (free — nodes already
   scored).
2. Run Stage 3, and for each of the 45 class pairs extract its full **bottleneck path**
   through the current MaxST — not just the single bottleneck edge.
3. Refine **every edge on those 45 paths** (union the edge sets, dedup) with the full
   §2.2 segment sampling. Weights can only *decrease*.
4. Re-run Stage 3. Repeat until the 45 bottleneck *paths* stop changing (expect 3–6
   rounds).

**Why the whole path, not just the bottleneck edge** (a correctness fix, not caution):
the extracted bottleneck edge is the *minimum-weight* edge on the current max-min path,
so refining it is necessary — but an *interior* path edge whose optimistic
`min(R(u),R(v))` overstates its true segment-min is never surfaced as "the bottleneck"
(it sits above the current minimum), so under the edge-only rule it would never be
refined, and its inflation would hold τ* above its true value indefinitely. τ* would
remain a *valid* upper bound (all weights ≥ true weights ⇒ max-min ≥ true max-min) but a
needlessly loose one, and — worse — the looseness is invisible, since §6.2's resolution
ablation only probes edges that *were* sampled. Refining the full path closes the gap:
across rounds every edge that can ever gate a reported τ* gets segment-sampled.

Because refinement only lowers weights, this converges from above and **each round's τ*
is a valid upper bound at every step.** Total extra R evaluations: still a few thousand
(45 paths × a handful of edges each × S interior points), not 170k.

---

## Stage 3 — Barrier extraction

This is single-linkage / union-find on similarity, and it gives all 45 pairs in one pass.

```
    sort edges by w DESCENDING
    for each edge (u,v,w):
        union(u,v)
        for each class pair (A,B) not yet joined:
            if any anchor of A and any anchor of B are now in the same component:
                τ*(A,B) = w                      # the current edge's weight
                mark (A,B) joined
```

`O(E log E)`. Output: a 10×10 symmetric matrix `τ*` and a dendrogram.

**Class-set semantics:** τ*(A,B) as written is the *max over endpoint choices* — the
best route between any anchor of A and any anchor of B. This is the right set-to-set
definition. Also record the *median* over anchor pairs as a robustness variant; if the
two differ wildly, one anchor is doing all the work and you should know.

**Note — ultrametricity is not a diagnostic here.** On a graph with min-edge bottleneck
weights, `τ*` is *exactly* ultrametric by construction (it is the cophenetic distance of
the single-linkage dendrogram). It therefore cannot detect estimator error. Use §7
instead. This corrects an earlier claim.

---

## Stage 4 — Routing and bridges

### 4.1 What is on the route

For each class pair, extract the actual bottleneck path through the MaxST. For each
node on it, record: nearest class anchor, and the provenance tag. Then ask what the
route is made of — own-pair interpolants only, filler samples, or *other* pairs'
interpolants (the interesting case).

### 4.2 Excision test — the surviving definition of routing

```
    for each C ∉ {A,B}:
        remove all nodes whose nearest anchor is class C
        recompute τ*_{¬C}(A,B)
        C is load-bearing for (A,B)  iff  τ*_{¬C}(A,B) < τ*(A,B) − margin
```

Cheap: no new R evaluations, just re-run union-find. 10 × 45 = 450 re-runs, seconds.

**Note this resolves theory question F2 affirmatively in the discrete setting.**
Ultrametricity forbids a two-hop route from *beating* the direct optimum, but excision
changes the feasible set, so it can and does lower τ*. Routing is measurable this way;
it was not measurable as a cost inequality.

**Nearest-anchor assignment** (used here and in §4.1): pixel-space L2 to the 320
anchors — the *same* metric the graph is built on, so "node belongs to C" means "sits
in C's pixel-space basin." Using a different metric here than in Stage 2 would make the
excised set incoherent with the graph it acts on.

**Calibrate `margin` per class C, not globally.** The number of nodes removed by
excising C, `n_C`, varies strongly across classes — a hub class is nearest to many
filler/interpolant nodes, a peripheral one to few — so a single global margin
over-flags load-bearing for large-basin classes (excising them disconnects the graph on
node-count alone) and under-flags for small ones. For each C: draw random node subsets
of size **`n_C`** (matched, not fixed), recompute τ* for each, and set
`margin_C = 95th percentile of the random τ* drop at that removal size`. Then C is
load-bearing for (A,B) iff `τ*_{¬C}(A,B) < τ*(A,B) − margin_C`. This isolates "C is a
semantic bridge" from "we removed a lot of nodes."

---

## Stage 5 — Controls and nulls

| # | Control | Question | Expected if the result is real |
|---|---|---|---|
| **5.1** | **Within-class barrier.** Split each class's 32 anchors into two disjoint halves; compute τ*(A, A′) using only same-class interpolant nodes for the between-region | Is there any cross-class barrier at all, above the within-class floor? | τ*(A,A′) ≫ τ*(A,B) for most pairs |
| **5.2** | **Shuffled-R null.** Randomly permute R values across nodes, keeping the graph fixed | Does the tree structure come from R, or from graph topology alone? | τ* matrix becomes structureless; Spearman vs true τ* ≈ 0 |
| **5.3** | **Degraded-node injection.** Add 2,000 G4-style degraded images as nodes | Does R correctly refuse to route through garbage? | τ* essentially unchanged. **If barriers rise, R is broken** — it is scoring degraded images as realistic |
| **5.4** | **Path-type ablation.** Rebuild with (a) slerp nodes only, (b) linear-blend nodes only, (c) both | Is the barrier an artifact of one seeding family? | τ* ordering stable across (a) and (c); (b) alone gives systematically lower τ* |
| **5.5** | **Anchor-shuffle null.** Reassign anchor class labels at random, keeping all nodes and R | Does the class structure come from the geometry or from the labelling? | τ* matrix loses structure |

5.1 and 5.3 are the two that most directly guard the headline. 5.3 is the one that
catches a broken R *after* Stage 0 passed.

---

## Stage 6 — Sensitivity (mandatory, not optional)

### 6.1 Coverage ablation
Recompute τ* with 25%, 50%, 75%, 100% of interpolant nodes (random subsets, 3 reps).
Plot τ*(A,B) vs node fraction for all 45 pairs. **Rising and not yet plateaued at 100%
means undersampled** — τ* is a lower bound and the true barrier is better. Report the
plateau status per pair.

### 6.2 Resolution ablation
Recompute with δ halved and doubled (i.e. more/fewer interior samples per edge).
**Falling with finer resolution means the segments hide dips** — τ* is an upper bound.
Report both bounds bracketing each barrier.

### 6.3 k sensitivity
k ∈ {10, 20, 40}. Larger k allows longer jumps; §2.2's length-proportional sampling
should absorb this. If τ* rises sharply with k, the segment sampling is too coarse.

---

## Stage 7 — Comparison graphs

Build four alternative 10×10 matrices and compare against τ*:

| Graph | Construction | Cost |
|---|---|---|
| `G_CLIP` | negative CLIP image-embedding centroid distance | minutes |
| `G_confusion` | symmetrised resnet50 confusion matrix on CIFAR-10 test | already have it |
| `G_fwdback` | **forward–backward transition matrix `T_σ`** (Sclocchi et al., PNAS 2025, arXiv:2402.16991): noise a class-A image to σ, denoise the full reverse ODE, record output class | ~1 GPU-hour |
| `G_pixel` | negative pixel-space class-centroid distance | seconds |

Metrics: Spearman ρ on the 45 off-diagonal entries; cophenetic correlation between
dendrograms; Robinson–Foulds distance between tree topologies.

**`G_fwdback` is the one that matters.** It is the cheap published alternative and the
project's main falsifier — see §8.

**σ-placement for `T_σ` is not free, and it is the same σ decision as Stage 1.** The
Sclocchi result is a *phase transition*: below a threshold σ the backward process
returns the same class (`T_σ ≈ I`, no cross-class structure) and above it the class
scrambles (`T_σ` → near-uniform). Informative `T_σ` structure lives in a narrow band
*around* the transition; run `T_σ` at ≥3 σ levels bracketing it and report the sweep,
not one σ. This also connects to Phase 1's empirical finding that any cross-class signal
lived at **low σ (0.5) and was gone by σ=2.0** — evidence the transition for CIFAR-10 /
this EDM sits low, which is *also* the σ at which the Stage-1 slerp interpolant nodes
should be seeded (the doc already fixes slerp σ=0.5; keep them consistent, and if the
Stage-0 realism ordering or `T_σ` says the transition is elsewhere, re-seed Stage 1 to
match before trusting τ*).

---

## Stage 8 — Predictions and falsifiers, committed in advance

Write these down before running Stage 3, and do not revise them afterward.

| # | Prediction | Falsifier | If falsified |
|---|---|---|---|
| **P1** | τ* has structure: the 45 values are not all equal, spread ≫ the shuffled-R null (5.2) | spread within null | No barrier structure at this σ / this R. Report and stop |
| **P2** | τ* is not a monotone function of `G_pixel` or `G_CLIP` (ρ < 0.8) | ρ > 0.9 with either | The tree is a re-derivation of embedding distance. Contribution collapses to method |
| **P3** | Vehicle classes {airplane, automobile, ship, truck} and animal classes form distinct dendrogram subtrees | they interleave, or the tree is a caterpillar | Generative connectivity is not taxonomic. Interesting, but a different paper |
| **P4** | At least one (A,B,C) triple shows a load-bearing C beyond the random-excision margin (§4.2) | zero triples | No bridge structure. τ* matrix stands alone as the result |
| **P5** | τ* differs from `T_σ` (ρ < 0.8) | ρ > 0.9 | **The project's main falsifier.** The 1-GPU-hour published method reproduces the expensive one; nothing is added by the barrier machinery except the tree |
| **P6** | Within-class barriers exceed cross-class (5.1) | they don't | Class basins are not resolved at this σ. Re-pick σ before anything else |

**P5 is the one to check first**, immediately after Stage 3 — before Stages 4–7. It is
the cheapest thing that can invalidate everything else.

---

## Compute budget

| Stage | R evals | Denoiser NFE | Wall clock (5090) |
|---|---|---|---|
| 0 — score validation | 2,000 × 5 candidates | ~10⁷ (Graham) | ~4 h |
| 1 — node generation | — | ~6×10⁵ (decode 15k interpolants + 2k samples) | ~20 min |
| 2 — node scoring | 17,280 | ~1.1×10⁷ (Graham) / ~10⁵ (SCOPED) | 2–5 h / minutes |
| 2 — edge refinement | ~5,000 | ~3×10⁶ | ~1 h |
| 3–5 — graph, barriers, controls, excision | 0 (except 5.3: +2,000) | ~1.3×10⁶ | ~30 min |
| 6 — sensitivity | 0 (reuses scores) | 0 | minutes |
| 7 — comparison, inc. `T_σ` | — | — | ~1 h |
| **Total** | ~27k | ~2.6×10⁷ | **~10 h with Graham, ~3 h with SCOPED** |

Roughly a quarter of the Phase-1 sweep, and it answers a question that sweep could not
address.

---

## What can go wrong, and what each failure means

**All routes go through the filler samples.** τ*(A,B) becomes "the lowest R needed to
hop between clusters of generator samples," identical for every pair. Diagnose via
§4.1 provenance. If it happens, either the filler is too dense (subsample it) or the
model's output manifold really is one connected blob at this σ — which is P1
falsified and a reportable finding.

**τ* is uniformly high (near max R).** The graph found a high-realism route between
every class pair. Either coverage is good and the manifold is genuinely well-connected,
or §6.2 shows the segments are hiding dips. Distinguishable.

**τ* is uniformly at the floor.** Every route dips into garbage. Check 5.1 first: if
within-class barriers are also at the floor, the interpolants themselves are broken
and the problem is Stage 1, not the geometry.

**One anchor dominates.** The max-over-endpoints definition lets a single weird image
set every barrier. Compare against the median variant in Stage 3.

**Stage 0 fails criterion 2 for every candidate.** No available score separates
manifold-aware mixtures from double exposures. This is the honest blocker, and it
would make realism estimation for near-OOD generative interpolants the actual research
contribution — a narrower but real paper.

---

## Order of operations

1. Stage 0. **Gate.** Do not proceed on a failed score.
2. Stage 1 + 2 (nodes, scores, graph).
3. Stage 3 (barriers) → **immediately test P5 against `T_σ`.**
4. If P5 holds: Stages 4, 5, 6, 7 in that order.
5. Only then decide whether continuous refinement (climbing-image string method on the
   3–5 most interesting pairs) is worth building.

---

## Rigor log — hardening pass (2026-09-04)

Changes made to the original draft, so the design's assumptions are traceable:

- **Citations verified, not assumed.** Graham (arXiv:2211.07740, CVPR 2023), SCOPED
  (arXiv:2510.01456) and EigenScore (arXiv:2510.07206, ICLR 2026) all confirmed real and
  correctly characterized; SCOPED and EigenScore both purpose-built for the near-OOD
  regime — the prior reason to expect §0.3 criterion 2 to be clearable. `G_fwdback`
  pinned to Sclocchi et al. PNAS 2025 (arXiv:2402.16991), and its **phase-transition σ**
  folded into the σ-placement decision it shares with Stage 1.
- **Lazy refinement (§2.3) corrected** from "refine the bottleneck edge ± 2-hop" to
  "refine the full bottleneck path each round" — the edge-only rule leaves inflated
  interior path edges permanently un-sampled, keeping τ* a *valid but silently loose*
  upper bound. Affordability is preserved.
- **Excision margin (§4.2) corrected** from a single global margin to a per-class,
  removal-size-matched null — a global margin miscalibrates load-bearing for hub vs.
  peripheral classes. Nearest-anchor metric pinned to pixel-L2 (graph-consistent).
- **Realism-score validation (§0.3) extended** with an off-midpoint smoothness gate:
  edges evaluate R at all `s∈(0,1)`, so a score validated only at `s=0.5` is
  under-validated for its actual use.
- **Locality sanity check (§2.1) added** — pixel-L2 is correct by construction but only
  useful if the node set is dense; the check catches the coverage failure early.
- **Linkage to Phase 1 and a reuse map (§0.4) added** — `τ*` is the RQ4/realism object
  independent of Phase-1 `C`; SCOPED reuses `utils/jvp.py` outright, which is what makes
  the SCOPED budget "minutes."

Unchanged and endorsed: the two-bias framing, the "don't sample p_σ / populate the
between-region by construction" node design, single-linkage/union-find barrier
extraction, the ultrametricity-is-not-a-diagnostic correction, the excision *test*
itself, the committed falsifiers (esp. P5-first), and the compute budget.
