# Understanding the Latent Space of Diffusion Models through the Lens of Riemannian Geometry

**Authors:** Yong-Hyun Park, Mingi Kwon, Jaewoong Choi, Junghyo Jo, Youngjung Uh (2023)

## Theoretical Abstract & Relevance to SEED Project

### Core Theoretical Contribution
This paper provides a rigorous geometric analysis of the latent space of diffusion models (DMs). Unlike GANs or VAEs, the diffusion latent space (the noise space $X_t$ at various timesteps) lacks obvious semantic structure, making it difficult to control or interpret. The authors overcome this by applying Riemannian geometry, specifically utilizing a **pullback metric**.

Because the latent space $X_t$ itself does not have a natural Euclidean metric that aligns with semantics, they pull back the Euclidean metric from an intermediate feature space $H$ (the bottleneck layer of the U-Net, which is known to be locally linear and semantic).
*   **Local Latent Basis:** By computing the Jacobian of the mapping from $X_t$ to $H$ ($J_x = \nabla_x h$) and performing Singular Value Decomposition (SVD), they extract the *local latent basis* (right singular vectors) and the *local tangent basis* (left singular vectors).
*   **Image Editing:** Moving along the principal components of this local basis at a specific timestep allows for semantically meaningful, zero-shot image editing (e.g., changing age, adding a beard). 
*   **Geometric Evolution:** They observe that the latent structure evolves over time: early diffusion timesteps (high noise) dictate low-frequency/coarse structural changes, while later timesteps dictate high-frequency/fine details. Furthermore, the local tangent spaces across different samples become increasingly dissimilar as the generative process progresses toward $t=0$.

### Relevance to SEED (Semantic Class Graph)
This paper is a **"Bedrock / Geometry lens"** reference for the SEED project.

1.  **Metric Definition (Methodology):** Park et al. establish the exact mathematical machinery—defining a pullback metric via the Jacobian of the neural network—required to treat the noise space of a diffusion model as a curved Riemannian manifold. SEED directly relies on this capability (the "score-Jacobian pullback metric") to compute distance and construct paths.
2.  **Local vs. Global Structure:** This paper focuses on *local* tangent spaces to find linear directions for semantic image editing (Program 1: Conditioning geometry / Local geometry). SEED expands on this by integrating these local metrics to find *global* geodesics (Program 2: Distribution geometry), transitioning from local image manipulation to mapping the global semantic connectivity between distinct classes.
3.  **Timestep Dependence:** The observation that geometric structure and tangent space homogeneity vary drastically across diffusion timesteps is critical for SEED. It implies that any routing or graph extraction must carefully select the timestep $\tau$ at which the geodesics are computed, as transition barriers and connectivity will look different at coarse (high noise) versus fine (low noise) scales.
