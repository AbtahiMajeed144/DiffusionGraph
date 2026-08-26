# SEED — Semantic Class Graph & Class-Routing in Generative Models

> **Purpose of this file.** This is the root spec for a Claude Code project. It defines the research question, the exact go/no-go experiment, and the phased build. It is written to be *executed*, not just read. The governing rule is encoded structurally: **a cheap experiment can kill this project, and it must be run before any scaling.** Do not build Phase 2+ until Phase 1's gate passes.
>
> **Confidence tags:** `[established]` peer-reviewed & reproduced · `[recent]` 2025–2026 credible · `[preprint — verify]` single-source, re-check before relying · `[assumption]` our design choice, revisable · `[open]` to be decided by experiment.

---

## 0. One-paragraph thesis

Class-conditional generative models place discrete classes in a continuous representation. When you construct a *realistic* (on-manifold) path between two distant classes, the path may not go directly — it may **route through a third real class** (e.g. cow → goat → dog), because that route stays in high-realism regions. If this happens systematically, the union of class manifolds has a recoverable connectivity structure: a **semantic class graph** implicit in the model's geometry. This project asks whether routing occurs, whether it is predictable, and whether the induced graph is a genuine, non-trivial object distinct from a classifier confusion matrix, a CLIP-distance graph, or a human taxonomy.

**Framing discipline (critical for publishability).** This is an **interpretability / science-of-generative-models** paper, *not* an interpolation-method paper. The contribution is a *finding about structure*, with the graph as its consequence. Lead every artifact (abstract, figures, README) with the finding, never with "a better interpolation algorithm." Rationale: the interpolation-geometry lane is active and near-saturated on the *how to make realistic paths* question; our whitespace is one level up — *what those paths reveal about class organization*.

---

## 1. The precise research questions

Ordered by importance, **not** by the order they appear in a paper. The paper will likely present RQ2 → RQ4 → RQ3, with RQ1 as supporting analysis.

- **RQ2 (central) — Routing.** Do realism-maximizing paths between distant classes traverse intermediate real classes more than chance, and more than naive straight-line paths?
- **RQ3 (highest ceiling) — The graph.** Can a class-connectivity graph be reconstructed from generative-model geometry such that its held-out shortest paths *predict* observed intermediate classes — and is this graph distinct from and more predictive than a confusion matrix / CLIP distances / WordNet?
- **RQ4 (methodological) — The objective.** Which path objective (shortest geodesic / highest-density / manifold-tangential / entropy-balanced principal curve) best satisfies "every point is a realistic member of *some* class"? This engages the open density-vs-manifold debate.
- **RQ1 (supporting) — Sharp transitions.** Along a realistic path, does classifier confidence stay saturated for one class then flip within a thin band, and does that band coincide with a geometric phase-boundary signature?

**Reframed central hypothesis (stronger than "routing must happen").** The scientifically honest question is *what determines* whether two classes are (A) directly connected, (B) connected via intermediate classes, or (C) separated by a generative barrier. Treat A/B/C as **three possible outcomes to be measured**, not a phenomenon to be confirmed. `[assumption]`

---

## 2. Three objects that must NOT be conflated

The single biggest reviewer attack. Keep these separate in code and prose:

1. **Conditioning interpolation** — moving the class condition `c(t) = (1−t)·c_A + t·c_B`. Cheap, but a continuous condition does *not* imply the line between embeddings is semantically meaningful.
2. **Distribution-geometry path** — a geodesic/string path through the learned distribution between *samples* of A and B, under a score-induced metric.
3. **Semantic trajectory** — the sequence of classes a path passes through.

**Design decision `[assumption]`:** the scientifically strong program studies **object 2** (sample-to-sample geometric paths) and *reads off* object 3, rather than assuming object 1 already encodes the structure. Conditioning interpolation is included only as a **baseline path type**, never as the primary object of study. This makes "we discovered structure from geometry" defensible rather than circular.

---

## 3. GO/NO-GO GATE — the first experiment (build this FIRST)

> **Nothing else in this project is built until this runs and reports.** Estimated effort: days, not weeks. This exists to falsify the project cheaply.

### 3.1 Question the gate answers
> Do reproducible intermediate-class peaks occur **preferentially on geometry-aware realistic paths** (vs straight-line paths), on a small dataset, robust across independent classifiers, and absent under a label-permutation control?

### 3.2 Setup
- **Dataset:** CIFAR-10 first (10 classes, all 45 unordered pairs). `[assumption]` Rationale: small enough to be exhaustive, semantic structure plausible (animals vs vehicles), classifiers are trivial to train.
- **Generator:** a pretrained/quickly-trained class-conditional diffusion model on CIFAR-10 (EDM-style is the default; a small class-conditional DDPM is an acceptable fallback). `[assumption]` Single RTX 5090 is ample.
- **Path types to compare (4):**
  1. Linear class-embedding interpolation (object 1 baseline).
  2. Slerp in noise space (prior-respecting baseline).
  3. A score-Jacobian **manifold-tangential geodesic** — *do not implement from scratch;* build on a released path optimizer (see §7 refs). `[assumption]`
  4. An entropy-balanced / string-method path (released string-method code if available).
- **Semantic evaluators (≥3, independent):** two classifiers of *different architectures* (e.g. a ResNet and a ViT) trained on CIFAR-10, plus one self-supervised embedding (e.g. DINO/CLIP nearest-class). Routing must survive all three. Rationale: classifier probability ≠ semantic membership; a single classifier's "goat" peak could be an artifact.

### 3.3 The measurement
For each ordered pair (A,B) and each path γ, sweep parameter t and record the full softmax `p(c | γ(t))` for each evaluator.

Define per-pair **routing strength**:
```
C(A,B) = max over t of [ max over c ∉ {A,B} of p(c | γ(t)) ]
```
A **routing event** = `C(A,B) > τ` (start τ = 0.5, report sensitivity to τ) that persists across all ≥3 evaluators and ≥3 seeds. Aggregate into a routing matrix `R[A,B]`.

### 3.4 The controls (these are what make it credible)
- **Straight-line contrast:** routing events must be *more frequent / stronger* on paths 3–4 than on paths 1–2. If straight lines route just as much, the geometry claim is empty.
- **Label-permutation negative control:** retrain/relabel with class IDs randomly permuted (image distributions unchanged). Any structure derived purely from arbitrary class *identifiers* must vanish. If "routing" persists under permutation, it's a metric/classifier artifact — **kill.**
- **Known-topology positive control (optional but strong):** construct a toy dataset with a *designed* semantic topology (e.g. interpolated-mixture classes with known adjacency) and check the method recovers it — identifiability evidence.

### 3.5 Gate decision
- **GO** if: routing events are reproducible across evaluators+seeds, preferentially on geometry-aware paths, and absent under permutation. → proceed to Phase 2.
- **PIVOT** if: routing is weak/absent but path objectives differ clearly in realism/chimera-rate. → the project becomes **RQ4-first** ("which objective minimizes off-manifold chimeras"), still publishable.
- **KILL** if: no routing, no objective separation, or structure survives permutation (artifact). → stop; write up the negative result if the controls are clean.

**Deliverable of the gate:** one figure (routing matrix `R` + example classifier-trajectory plots for 2–3 routed pairs) and a one-page decision memo.

---

## 4. Problem breakdown (what to build, in order)

Each phase has an entry gate. Do not start a phase until the prior gate passes.

### Phase 1 — Gate (§3). Exhaustive CIFAR-10 routing study + controls.
### Phase 2 — Systematicity. CIFAR-100: is the routing matrix *structured* (clusters, hierarchy) rather than random? Enough classes to ask graph questions.
### Phase 3 — The graph & held-out prediction. Build the class graph from Phase-2 routing/geodesic costs; test **held-out**: hide some pairs, predict their intermediate classes from the graph, verify against actual generated paths. Compare graph vs confusion-matrix / CLIP / WordNet as predictors of routing.
### Phase 4 — Objective study (RQ4). Systematically vary path objective; measure realism (FID/LPIPS + realism index) and chimera rate; report how the induced graph changes with objective. **"Different notions of on-manifold induce different class-transition graphs"** is a strong result.
### Phase 5 (optional) — Fine-grained / scale. CUB or a small ImageNet subset, only if 1–4 succeed. **Do not start at ImageNet.**

---

## 5. Method details & resolved design choices

- **Metric / paths.** Score-Jacobian pullback metric on a pretrained diffusion model (score gives density gradients for free). `[assumption]` **Never build full Jacobians** — use JVP/VJP, Hutchinson trace estimates, low-rank eigensolvers, and operate in latent/reduced resolution. Prefer building on a *released* geodesic/string optimizer over writing one (feasibility unlock).
- **Class-to-class distance — NOT centroids.** Define connectivity over **sample sets / modes**, e.g. `D(A,B) = min over x~A, y~B of pathcost(γ_{x→y})`, or a low-percentile transition cost. `[assumption]` Rationale: multimodal classes make centroid paths cross empty regions and fabricate false barriers.
- **Graph object naming.** Call it a **generative connectivity graph** / **model-induced class-adjacency graph**, not a "semantic graph." It may encode the model's *perceptual* organization (shape/texture/background) rather than human taxonomy — which is a *more* interesting interpretability finding, framed honestly.
- **Realism is not one objective.** Track separately: likelihood, density, distance-to-manifold, classifier confidence, perceptual (LPIPS/FID), realism index. Do not collapse them; RQ4 is about their disagreement.

---

## 6. Evaluation & anti-confounding checklist

- Routing must survive **≥3 independent evaluators** (2 classifier architectures + 1 SSL embedding). `[established practice]`
- **Held-out graph prediction**, not in-sample fit (avoids the circularity: build graph on some pairs → predict others → verify). `[established practice]`
- **Label-permutation control** for identifiability. `[established practice]`
- **Baseline graphs** to beat as routing predictors: classifier confusion matrix, CLIP-distance graph, WordNet/taxonomy (where available), raw feature-space distance.
- **Multiple path objectives** compared, not one (RQ4 is a feature, not an afterthought).
- Report τ-sensitivity and seed variance for every routing claim.

---

## 7. Foundational references (sequential; verify identifiers before citing)

> Read *to the gate*, then stop. Phase-3 papers are the novelty threats — read adversarially, asking "does this already make my routing/graph claim?"

**Bedrock (read once):**
- Ho, Jain, Abbeel. *Denoising Diffusion Probabilistic Models.* NeurIPS 2020. arXiv:2006.11239. `[established]`
- Song et al. *Score-Based Generative Modeling through SDEs.* ICLR 2021. arXiv:2011.13456. `[established]`
- (opt.) Lipman et al. *Flow Matching for Generative Modeling.* ICLR 2023. arXiv:2210.02747. `[established]`

**Geometry lens (read carefully):**
- Arvanitidis, Hansen, Hauberg. *Latent Space Oddity.* ICLR 2018. arXiv:1710.11379. `[established]`
- Park et al. *Understanding the Latent Space of Diffusion Models through the Lens of Riemannian Geometry.* NeurIPS 2023. arXiv:2307.12868. `[established]`
- (back-ref) Shao et al. *The Riemannian Geometry of Deep Generative Models.* CVPR-W 2018. arXiv:1711.08014. `[established]`
- (back-ref) Chen et al. *Metrics for Deep Generative Models.* AISTATS 2018. arXiv:1711.01204. `[established]`

**Direct competition — read adversarially, highest priority:**
- Saito, Matsubara. *Be Tangential to Manifold: Discovering Riemannian Metric for Diffusion Models.* arXiv:2510.05509, 2025. `[preprint — verify]`
- Moreau et al. *Probing the Geometry of Diffusion Models with the String Method.* arXiv:2602.22122, 2026. `[preprint — verify authors/number]`

**Novelty threats — read to differentiate:**
- *Encoded Prior Sliced Wasserstein AutoEncoder (EPSWAE).* arXiv:2010.01037, 2020 — the un-measured 7→9→4 routing aside. `[preprint — verify authors]`
- Jaquier, Rozo et al. *Bringing Motion Taxonomies to Continuous Domains via GPLVM on Hyperbolic Manifolds.* arXiv:2210.01672, 2022 — *supplied* taxonomy (contrast: we *discover*). `[recent]`

**Read only after the gate passes:**
- Lobashev et al. *Hessian Geometry of Latent Space in Generative Models.* ICML 2025. arXiv:2506.10632 — RQ1 phase-boundary theory. `[recent]`
- Struski et al. *Feature-Based Interpolation and Geodesics in the Latent Spaces of Generative Models.* IEEE TNNLS 2023. arXiv:1904.03445 — realism index metric. `[established]`
- Zhu et al. *Diff-Mix.* CVPR 2024. arXiv:2403.19600 — inter-class mixing (related work). `[established]`
- He et al. *AID: Attention Interpolation of Text-to-Image Diffusion.* NeurIPS 2024. arXiv:2403.17924 — condition interpolation (related work). `[established]`

---

## 8. Risks, kills, and honest caveats

- **Effect-size risk (main internal risk).** Routing may be weak or dataset/model-dependent. *Mitigation:* the gate tests this on day one; the RQ4 pivot survives a null.
- **Connectivity may be conditional, not universal** (topology varies with model/training). That is itself a measurable, publishable question — not a defeater.
- **Novelty risk (main external risk).** The Phase-3 preprints are close on the *geometric* half. *Mitigation:* read them first; position as "one level up" (routing + graph discovery), never as "nobody studies connectivity in diffusion."
- **Confounding.** Classifier artifacts, metric artifacts, centroid artifacts — all addressed by the §6 checklist; skipping any of it invites rejection.
- **Preprint instability.** Several load-bearing refs are un-peer-reviewed 2025–2026 preprints; track published versions, verify identifiers before submission.
- **Strategic caveat (not a technical risk).** This project does **not** use the author's Bengali/Indic edge; it competes on geometric/experimental rigor. A deliberate choice, worth revisiting.

---

## 9. What "done" looks like (target contributions)

1. **Phenomenon:** routing quantified, reproducible, geometry-preferential, artifact-controlled.
2. **Predictability:** a held-out-validated class graph whose shortest paths predict intermediate classes.
3. **Distinctness:** `G_generative ≠ G_CLIP ≠ G_confusion ≠ G_taxonomy`, and `G_generative` predicts actual transitions best — the strongest outcome.
4. **Objective insight:** which realism objective yields the most on-manifold transitions, and how the graph depends on it.

Target genre/venue: interpretability / science-of-generative-models → NeurIPS / ICLR / ICML main track. Finding-first framing is mandatory.

---

## 10. Immediate next actions for the Claude Code agent

1. Scaffold the repo: `data/`, `models/` (pretrained CIFAR-10 class-conditional diffusion), `paths/` (4 path constructions), `eval/` (≥3 classifiers/embeddings + trajectory logger), `analysis/` (routing matrix, controls), `results/`.
2. Implement the **gate (§3)** end-to-end on CIFAR-10 before anything else. Wire in JVP/VJP score-Jacobian utilities; prefer a released geodesic/string optimizer.
3. Produce the gate deliverable: routing matrix figure + trajectory plots + one-page decision memo (GO / PIVOT / KILL against §3.5).
4. **Halt for human review of the gate result.** Do not scaffold Phase 2+ until the gate passes.
