"""Download both pretrained EDM CIFAR-10 checkpoints we need.

We use TWO checkpoints, deliberately, not one -- this is the load-bearing
fix for CFG/conditioning contamination (Strategic_Blind_Spots_Analysis.md
#2), stronger than just pinning guidance weight w=1.0 on a single model:

  - edm-cifar10-32x32-cond-vp.pkl    Object 1 (SEED §2): conditioning
                                      interpolation. ONLY path type 1
                                      (linear_condition) uses this.
  - baseline-cifar10-32x32-uncond-vp.pkl   Object 2 (SEED §2): distribution
                                      geometry. Path types 2/3/4 use this --
                                      the marginal score over the whole data
                                      distribution, with no class label to
                                      bias which mode a path is pulled
                                      toward. This is what makes "we
                                      discovered structure from geometry"
                                      defensible rather than circular (SEED
                                      §2's own framing requirement).
"""
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diffusiongraph.config import CHECKPOINTS_DIR

BASE_URL = "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained"
CHECKPOINTS = [
    "edm-cifar10-32x32-cond-vp.pkl",            # EDM's own config-F conditional model
    "baseline/baseline-cifar10-32x32-uncond-vp.pkl",  # config-A baseline, re-imported from
                                                 # Song et al.'s score_sde_pytorch (VP) --
                                                 # EDM does not host a downloadable
                                                 # config-F *un*conditional CIFAR-10
                                                 # checkpoint, only the training recipe
                                                 # to reproduce one (README.md L204-207).
]


def _reporthook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    pct = min(100, downloaded * 100 // total_size) if total_size > 0 else 0
    mb = downloaded / 1e6
    total_mb = total_size / 1e6
    print(f"\r{pct:3d}%  {mb:7.1f} / {total_mb:.1f} MB", end="", flush=True)


def main():
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    for name in CHECKPOINTS:
        dest = CHECKPOINTS_DIR / Path(name).name  # flatten baseline/ prefix locally
        if dest.exists():
            print(f"Already present: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
            continue
        url = f"{BASE_URL}/{name}"
        print(f"Downloading {url}\n  -> {dest}")
        urllib.request.urlretrieve(url, dest, reporthook=_reporthook)
        print(f"\nDone: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
