# Third-party code and licenses

`references/` is git-ignored and populated by `scripts/setup_references.sh`. We
build *on top of* these (JVP/VJP utilities, geodesic-curve scaffolding, evaluator
architectures) rather than vendor them into our own tracked history. Licenses below
govern anything in `diffusiongraph/` that is a direct adaptation — cite accordingly
in any paper/README.

| Repo | Commit | License | We use it for |
|---|---|---|---|
| [NVlabs/edm](https://github.com/NVlabs/edm) | `008a4e5` | CC BY-NC-SA 4.0 (**non-commercial**) | Pretrained class-conditional CIFAR-10 diffusion model (the generator under study). Checkpoint download separate — see `scripts/download_edm_checkpoint.py`. |
| [enkeejunior1/Diffusion-Pullback](https://github.com/enkeejunior1/Diffusion-Pullback) | `859c012` | Apache-2.0 | Reference for the Jacobian-pullback-metric technique (Park et al. NeurIPS'23), adapted to CIFAR-10/EDM rather than reused directly (targets CelebA-HQ/SD). |
| [MachineLearningLifeScience/stochman](https://github.com/MachineLearningLifeScience/stochman) | `b0acd1e` | Apache-2.0 | Base for the geodesic/curve-energy optimizer used in path-type 3 (score-Jacobian tangential geodesic). |
| [yang-song/score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch) | `cb1f359` | Apache-2.0 | Score-function API reference; backup generator path. |
| [kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar) | `49b7aa9` | MIT | ResNet18 architecture, evaluator #1. |
| [omihub777/ViT-CIFAR](https://github.com/omihub777/ViT-CIFAR) | `ab9043e` | MIT | ViT-small-from-scratch architecture, evaluator #2 (independent architecture family, per SEED §3.2). |
| [openai/CLIP](https://github.com/openai/CLIP) | `d05afc4` | MIT | Zero-shot nearest-class embedding, evaluator #3 (independent SSL signal, per SEED §3.2). |

**Note on the EDM license:** CC BY-NC-SA is non-commercial and share-alike. Fine for
academic research and publication; would need revisiting before any commercial use
of derived code/models.

**No code found for our two closest methodological references** — checked directly
against their arXiv pages (no repo, no supplementary link):
- Saito & Matsubara, *Be Tangential to Manifold* (arXiv:2510.05509)
- Moreau et al., *Probing the Geometry of Diffusion Models with the String Method* (arXiv:2602.22122)

Path-types 3 and 4 in our gate are therefore original implementations of their
published math, built on `stochman`'s curve machinery — not adaptations of a release.
