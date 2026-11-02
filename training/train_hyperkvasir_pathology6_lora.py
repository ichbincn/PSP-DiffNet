"""Train the shared class-conditioned SD1.5 LoRA adapter on Raw pathology6.

The supplied LoRA trainer is reused unchanged; this module only binds the
frozen six-class vocabulary/prompts and the Raw endoscopy preprocessor.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.hyperkvasir_pathology import (  # noqa: E402
    HYPERKVASIR_PATHOLOGY6_CLASS_ORDER,
    HYPERKVASIR_PATHOLOGY6_PROMPTS,
)
from utils.endoscopy import preprocess_endoscopy_pil  # noqa: E402
import training.train_mmotu_lora as trainer  # noqa: E402


trainer.MMOTU_CLASS_ORDER = list(HYPERKVASIR_PATHOLOGY6_CLASS_ORDER)
trainer.PROMPTS = dict(HYPERKVASIR_PATHOLOGY6_PROMPTS)
trainer.preprocess_ultrasound_pil = preprocess_endoscopy_pil
trainer.PREPROCESSING_DESCRIPTION = (
    "Raw EXIF-transposed RGB; aspect-preserving square resize-pad; horizontal flip only"
)


if __name__ == "__main__":
    sys.exit(trainer.main())
