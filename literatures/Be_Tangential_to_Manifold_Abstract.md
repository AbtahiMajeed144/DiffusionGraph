# Be Tangential to Manifold: Discovering Riemannian Metric for Diffusion Models

**Authors:** Shinnosuke Saito and Takashi Matsubara (2025/2026)

## Theoretical Abstract & Relevance to SEED Project

### Core Theoretical Contribution
This paper introduces a training-free Riemannian metric on the noise space of pre-trained diffusion models. Unlike classical models (e.g., VAEs, GANs), diffusion models operate in a noise space with the same dimensionality as the data space, lacking a low-dimensional latent space to parameterize the data manifold. 
Prior approaches used density-based metrics (which tend to over-smooth by pushing paths to high-density regions) or relied on heuristics. 

The authors propose a Riemannian metric derived directly from the **Jacobian of the score function**: $J_{x_t} = \nabla_{x_t} s_\theta(x_t, t)$.
*   **Spectral Separation:** The Jacobian's spectral structure naturally separates tangent and normal directions of the data manifold. Tangent directions correspond to small singular values, while normal directions correspond to large singular values.
*   **The Metric:** Defined as the pullback of the Euclidean metric through the score function: $g_{x_t}(v, w) = v^\top J_{x_t}^\top J_{x_t} w$. This assigns a high cost to movement along normal directions and a low cost along tangent directions.
*   **Geodesics:** By minimizing the energy functional $\int_0^1 \| J_{\gamma(u)} \gamma'(u) \|_2^2 du$, the geodesic paths are encouraged to stay *tangential* to the data manifold, moving parallel to it rather than drifting into the highest-density regions.

### Relevance to SEED (Semantic Class Graph)
This paper is **directly foundational** to the SEED project, fulfilling the geometric requirements outlined in the project specifications:

1.  **RQ4 (Methodological Objective):** The paper provides the theoretical and computational foundation for the "score-Jacobian manifold-tangential geodesic" path type. It solves the problem of constructing paths that remain realistic without merely collapsing into high-density "over-smoothed" modes (which might distort intermediate semantic routing).
2.  **Feasibility of Score-Jacobian Computation:** The SEED spec states *"Never build full Jacobians — use JVP/VJP"*. This paper validates that approach, showing that $G_{x_t}$ computations can be tractably implemented using Jacobian-vector products (via finite differences or automatic differentiation) to approximate $\Delta s$.
3.  **Distribution Geometry Path vs. Condition Geometry:** This metric operates purely on the distribution geometry (noise space) rather than condition space. It allows the SEED project to establish unbiased, geometry-aware realistic paths (Object 2 in SEED) to measure semantic trajectories (Object 3).
4.  **Avoiding Density Bias:** A major risk in interpolating across classes is falling into unnatural chimeras or being drawn entirely to a dominant density mode. By prioritizing the tangent space over pure density, this geodesic method is a prime candidate for uncovering the true "generative connectivity graph."
