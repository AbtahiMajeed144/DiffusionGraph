# Probing the Geometry of Diffusion Models with the String Method

**Authors:** Elio Moreau, Florentin Coeurdoux, Grégoire Ferre, Eric Vanden-Eijnden (2026)

## Theoretical Abstract & Relevance to SEED Project

### Core Theoretical Contribution
This paper introduces a robust framework for probing the geometry of learned distributions in diffusion models using the **string method**, originally developed in computational chemistry. By evolving entire curves (strings) of samples rather than individual points under the learned score function, the method can compute continuous pathways between two generated samples directly on pretrained models, without requiring an explicit, time-independent energy function.

The authors explore three regimes of string dynamics:
1.  **Pure Generative Transport ($\gamma_t=0$):** Relies purely on the learned velocity field, producing continuous paths that lack intrinsic geometric grounding.
2.  **Minimum Energy Paths (MEPs) ($\gamma_t \gg 1, T=0$):** Gradient-dominated dynamics that maximize likelihood. In high dimensions, this exposes a **"likelihood-realism paradox"**: MEPs traverse likelihood maxima that lie outside the *typical set* where actual data mass concentrates. Consequently, MEPs produce unrealistic, "cartoonish" intermediate images.
3.  **Finite-Temperature Principal Curves ($\gamma_t \gg 1, T>0$):** By incorporating temperature to account for entropy, the string identifies *principal curves*—self-consistent paths that balance energy and entropy. These curves remain within the typical set and successfully produce realistic morphing sequences.

### Relevance to SEED (Semantic Class Graph)
This paper is classified as **"Direct Competition — Read Adversarially"** in the SEED project.

1.  **Direct Tool for Path Construction (RQ4):** The finite-temperature string method perfectly describes the "entropy-balanced / string-method" path mentioned as baseline #4 in SEED's experimental setup. It solves the exact problem of finding paths that satisfy "every point is a realistic member of some class" without falling into the "cartoonish" trap of pure density-maximizing MEPs.
2.  **The "Likelihood vs. Realism" Paradox:** The finding that MEPs (highest density) produce cartoon chimeras while principal curves (entropy-balanced) produce realistic interpolations is the exact theoretical tension being evaluated in SEED's RQ4.
3.  **Differentiation Strategy (Crucial):** While Moreau et al. provide the *machinery* to compute transition pathways and study modal connectivity, their focus is on the interpolation algorithm and the physics of the transitions. SEED differentiates by building *on top* of this machinery. Instead of just proving the path is realistic, SEED's goal is to **label the intermediate semantic states**, map out **systematic class routing**, and ultimately discover a global **generative connectivity graph**. The novelty of SEED lies in the extraction of predictable macro-structure (the graph) from the continuous geometry that Moreau et al. formalized.
