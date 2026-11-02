"""ConvNeXt-Tiny downstream evaluation adapter for HyperKvasir pathology6.

The training/evaluation implementation is shared with the frozen MMOTU
comparison runner.  This thin adapter only swaps the dataset vocabulary and
the image preprocessing function; it intentionally keeps the same
ImageNet-pretrained ConvNeXt-Tiny, optimizer, sampler, metric and checkpoint
selection code for every method.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.hyperkvasir_pathology import (
    HYPERKVASIR_PATHOLOGY6_CLASS_ORDER,
    HYPERKVASIR_PATHOLOGY6_MINORITY,
)
from utils.endoscopy import preprocess_endoscopy_pil

import training.run_mmotu_convnext as runner


runner.MMOTU_CLASS_ORDER = list(HYPERKVASIR_PATHOLOGY6_CLASS_ORDER)
runner.MMOTU_MINORITY = list(HYPERKVASIR_PATHOLOGY6_MINORITY)
runner.preprocess_ultrasound_pil = preprocess_endoscopy_pil
runner.AUGMENTATION_DESCRIPTION = (
    "Raw EXIF-transposed RGB; aspect-preserving square resize-pad; "
    "HFlip0.5; affine +/-10deg/3%/0.95-1.05; brightness0.10 contrast0.12"
)
runner.NO_AUGMENTATION_DESCRIPTION = (
    "Raw EXIF-transposed RGB; aspect-preserving square resize-pad; "
    "no stochastic augmentation"
)


if __name__ == "__main__":
    # argparse is intentionally owned by the shared runner so all command
    # line options and output manifests remain identical across datasets.
    sys.exit(runner.main())
