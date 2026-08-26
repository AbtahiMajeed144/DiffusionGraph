# Encoded Prior Sliced Wasserstein AutoEncoder for Learning Latent Representations of Data

**Authors:** Sanjukta Krishnagopal and Jacob Bedrossian (2020/2021)

## Theoretical Abstract & Relevance to SEED Project

### Core Theoretical Contribution
This paper aims to improve upon standard Variational Autoencoders (VAEs) which typically assume a simplistic Gaussian prior that fails to capture the true topological and geometric structure of complex data manifolds. The authors propose the **Encoded Prior Sliced Wasserstein AutoEncoder (EPSWAE)**.

The architecture involves a standard autoencoder paired with an additional *prior-encoder network* that explicitly learns to embed the data manifold's geometry into an unconstrained prior distribution. 
To train this, the authors utilize a **Nonlinear Sliced Wasserstein (NSW)** distance to measure the divergence between the learned prior and the posterior distributions, allowing for arbitrary prior shapes without the strict constraints of KL-divergence.

Crucially, to perform interpolations, they introduce a **graph-based network-geodesic algorithm**:
1. They sample points from the learned prior.
2. A graph is constructed where edges are formed based on distance thresholds scaled by local $k$-nearest neighbor distances. This promotes connections through dense regions.
3. Dijkstra's algorithm is used to find the shortest path (minimum energy) between two points on this graph.
This ensures interpolations stay on the high-density manifold rather than crossing empty Euclidean space.

### Relevance to SEED (Semantic Class Graph)
This paper is highly relevant to the SEED project as a **"Novelty Threat / Contrastive Reference"**. 

1. **Routing Phenomenon & Novelty Threat:** The paper explicitly explores non-linear transitions through dense regions of latent space. As noted in the SEED specification, the EPSWAE paper demonstrates interpolations like 7 → 9 → 4 on MNIST. This is a direct example of semantic routing, which is the core phenomenon SEED aims to quantify. SEED must differentiate itself by emphasizing the *systematic, predictable discovery of the global semantic connectivity graph* rather than just presenting another algorithm to perform manifold interpolation.
2. **Path Construction Method (RQ4):** The network-geodesic algorithm proposed in this paper (Dijkstra on density-thresholded prior samples) provides a concrete example of an **entropy-balanced / density-maximizing path**. In SEED's RQ4 (the objective study), contrasting paths that prioritize density (like EPSWAE) against paths that prioritize being tangential to the manifold (like the score-Jacobian metric) is critical to understanding how different definitions of "on-manifold" alter the induced class-transition graph.
3. **Graph vs. Path:** EPSWAE uses a graph *internally* to approximate a continuous path (geodesic). SEED, on the other hand, extracts continuous paths to discover a *semantic graph* of class relationships. This structural inversion in how the graph is utilized is a key point of differentiation.
