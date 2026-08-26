# Latent Space Oddity: On the Curvature of Deep Generative Models

**Authors:** Georgios Arvanitidis, Lars Kai Hansen, Søren Hauberg (2018)

## Theoretical Abstract & Relevance to SEED Project

### Core Theoretical Contribution
This paper challenges the common assumption that the latent space of deep generative models (like VAEs) is a flat, Euclidean space. Because the generator function mapping latent points to the input space is highly non-linear, it severely distorts the latent space. Consequently, standard Euclidean distances in the latent space are physically meaningless and do not reflect true distances on the data manifold.

To correct this, the authors mathematically formalize the latent space as a curved Riemannian manifold. 
*   **The Metric:** The distortion is characterized by a Riemannian metric defined via the Jacobian of the generator function: $M_z = J_z^\top J_z$.
*   **Stochastic Generators:** For stochastic generators (like standard VAEs), the expected metric incorporates both the Jacobian of the mean function and the Jacobian of the variance function. 
*   **Variance Regularization:** The authors note that standard neural networks provide poor variance estimates in regions far from the training data. If variance does not increase outside the data support, geodesics might cross empty space. By enforcing high variance away from data (using an RBF network for precision), the metric tensor creates a "wall" of high cost around the data support.
*   **Geodesics:** When interpolating by computing the shortest path (geodesic) using this metric, the path naturally avoids regions of high uncertainty (empty space) and follows the data manifold.

### Relevance to SEED (Semantic Class Graph)
This is a **foundational "Geometry lens" reference** for the SEED project.

1.  **Validating Object 2 (Distribution-Geometry Path):** This paper establishes the core theoretical justification for why SEED studies Object 2 (distribution-geometry paths) instead of Object 1 (conditioning interpolation). A straight line in a latent space is geometrically invalid because it ignores the curvature induced by the generative model; it easily crosses low-density barriers, generating unrealistic chimeras.
2.  **Routing Necessity:** By proving that properly constructed geodesics avoid regions of high uncertainty/variance, this paper conceptually validates SEED's central hypothesis: that realism-maximizing paths between distant classes will traverse intermediate real classes (high density/confidence regions) rather than cutting through empty space.
3.  **Generalization to Diffusion:** While this paper focuses on VAEs and their explicit generators, the underlying Riemannian geometry framework directly paved the way for later works (like Saito & Matsubara) that apply score-Jacobian metrics to diffusion models, providing the essential mathematical toolkit for SEED's path construction.
