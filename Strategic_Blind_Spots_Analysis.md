# Strategic Blind Spots & Unique Angles for the SEED Project

After synthesizing the foundation papers against the `SEED_semantic_class_graph.md` and `Analysis_of_gpt.md` specifications, I have identified several critical blind spots and unique theoretical angles that have been overlooked. Addressing these before diving into the core implementation will save significant wasted effort and strengthen the scientific novelty of the project.

## 1. The Timestep Dependency of the Graph (The $\tau$ Variable)

**The Blind Spot:** Both the SEED document and the GPT analysis implicitly treat the "semantic class graph" as a static, singular object. 
**The Reality:** As shown in *Understanding the Latent Space of Diffusion Models*, the geometry of the latent space evolves drastically over the generative process. 
*   **High Noise ($t \approx T$):** The metric is dominated by low-frequency signals. The semantic graph at this level will connect classes based on coarse structural similarity (e.g., "blob with four legs" connects Dog and Cat).
*   **Low Noise ($t \approx 0$):** The metric is dominated by high-frequency signals. The graph here will connect classes based on texture and fine details.

**The Unique Angle:** The semantic graph is not a static object $G$, but a **dynamic filtration $G(\tau)$**. You should not compute geodesics across the entire integration path $T \to 0$. Instead, the experiment should measure the routing phenomenon at specific, fixed timesteps $\tau$. Comparing the "coarse graph" to the "fine graph" could be a massive standalone scientific contribution.

## 2. The Classifier-Free Guidance (CFG) Distortion Effect

**The Blind Spot:** None of the project documents mention the role of Classifier-Free Guidance (CFG). Modern diffusion models rely heavily on CFG ($w > 1$) to generate distinct, high-fidelity classes.
**The Reality:** Recent literature shows that CFG significantly distorts the natural geometry of the latent space. It causes "norm amplification" and extrapolation that pushes samples off the natural data manifold. 
**The Unique Angle:** If you use a guided score $\tilde{s}_t = s_{uncond} + w(s_{cond} - s_{uncond})$ to compute your score-Jacobian metric, the resulting graph will not reflect the natural topology of the data. It will reflect the artificial, extrapolated barriers created by CFG. 
*   **Recommendation:** Phase 1 (CIFAR-10 Gate) MUST be run on an **unconditional model** or a conditional model with $w=1.0$. If you want to study CFG, formulate it as a specific ablation: "How does CFG sever natural semantic connections?"

## 3. The "Multi-Modal Class" Assumption

**The Blind Spot:** The routing matrix in SEED assumes that a "Class" is a single, well-defined node. 
**The Reality:** Real-world classes are highly multimodal. For example, the class "Car" contains "side-profile sedans" and "front-facing trucks". 
**The Unique Angle:** A geodesic from "Dog" to "Car" might route through "Cat" if we start from a side-profile Dog, but it might route through "Train" if we start from a front-facing Dog. If you just average paths between random samples of Class A and Class B, you might wash out the true topology. 
*   **Recommendation:** Do not compute class-to-class distance directly. Compute **mode-to-mode** distances (using k-means clustering on the start/end embeddings) and then aggregate them into a class-level hypergraph.

## 4. Computational Intractability of the Finite-Temperature String Method

**The Blind Spot:** SEED's RQ4 proposes comparing the purely tangential (score-Jacobian) paths against the "entropy-balanced / string method" (Principal Curves from Moreau et al.).
**The Reality:** Computing a finite-temperature string method across 70 images requires spawning stochastic walkers for each image, simulating SDEs at very fine timesteps ($\Delta t = \mathcal{O}(\gamma_t^{-2})$), and re-parametrizing constantly. Doing this over enough pairs to build a CIFAR-10 routing matrix will require astronomical compute. 
**The Unique Angle:** You need an early-exit proxy. Instead of running full Principal Curve optimization for every path, use the pure Generative Transport ($\gamma = 0$) or a highly discretized Score-Jacobian path as a fast proxy to identify *potential* routings. Only deploy the expensive finite-temperature string method on the most interesting edges (e.g., to prove that the 7->9->4 route survives entropy-balancing).

## 5. The "Chimera vs. True Mode" Evaluator Bias

**The Blind Spot:** You plan to use a pre-trained ResNet/CLIP to evaluate if an intermediate point on the path belongs to Class C.
**The Reality:** An MEP (Minimum Energy Path) often creates "cartoonish" chimeras (as noted by Moreau et al.). A chimera might have the texture of a cat and the shape of a dog. Neural networks have known inductive biases (e.g., ResNets are heavily biased toward texture). 
**The Unique Angle:** Your evaluators might confidently classify a chimera as a "Goat" simply because it has goat-like fur, even if the geometry makes no sense. The label-permutation negative control (mentioned in SEED) is good, but you should also add a **frequency-domain control** (e.g., checking if the intermediate images have typical natural image spectra) to ensure you aren't just discovering adversarial examples in the classifier's feature space.
