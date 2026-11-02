"""Stage wrapper for the six-class HyperKvasir pathology-only Raw line.

The repository's diffusion/A3/A4 implementations were originally written for
MMOTU.  This thin wrapper patches their dataset-specific registries before
dispatching, so the model code and its algorithmic definitions remain shared.
The public source bundle includes only the six-class pathology workflow.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.hyperkvasir_pathology import (
    HYPERKVASIR_PATHOLOGY6_CLASS_ORDER as CLASSES,
    HYPERKVASIR_PATHOLOGY6_CLASS_SLUGS as SLUGS,
    HYPERKVASIR_PATHOLOGY6_MINORITY as MINORITY,
    HYPERKVASIR_PATHOLOGY6_PROMPTS as PROMPTS,
)
from utils.endoscopy import preprocess_endoscopy_pil, preprocess_endoscopy_prior


NEGATIVE = (
    "illustration, cartoon, painting, text overlay, watermark, duplicated anatomy, "
    "severe geometric distortion, checkerboard artifact, extreme overexposure, "
    "underexposure, blurred mucosa, unreadable symbols"
)


def patch_lora(module) -> None:
    module.MMOTU_CLASS_ORDER = CLASSES
    module.PROMPTS = PROMPTS
    module.preprocess_ultrasound_pil = preprocess_endoscopy_pil
    module.PREPROCESSING_DESCRIPTION = (
        "Raw RGB endoscopy screenshot; aspect-preserving resize-pad; horizontal flip only; no ROI/mask"
    )


def patch_classifier(module) -> None:
    module.MMOTU_CLASS_ORDER = CLASSES
    module.MMOTU_MINORITY = MINORITY
    module.preprocess_ultrasound_pil = preprocess_endoscopy_pil
    module.AUGMENTATION_DESCRIPTION = (
        "Raw RGB endoscopy screenshot; aspect-preserving resize-pad; HFlip0.5; "
        "affine ±10deg/3%/0.95-1.05; brightness0.10 contrast0.12; scratch ConvNeXt-Tiny"
    )


def patch_generation(module) -> None:
    module.MMOTU_CLASS_ORDER = CLASSES
    module.MMOTU_CLASS_SLUGS = SLUGS
    module.PROMPTS = PROMPTS
    module.NEGATIVE_PROMPT = NEGATIVE
    module.preprocess_ultrasound_pil = preprocess_endoscopy_pil
    module.preprocess_ultrasound_mask = preprocess_endoscopy_prior
    module.PRIOR_SOURCE_DESCRIPTION = (
        "train-source deterministic Level-3 multi-scale mucosal-gradient soft prior; no mask/annotation"
    )


def patch_quality(module) -> None:
    module.MMOTU_CLASS_ORDER = CLASSES
    module.preprocess_ultrasound_pil = preprocess_endoscopy_pil
    module.REAL_REFERENCE_DESCRIPTION = "fixed HyperKvasir pathology6 Raw real train; generated minority classes only"
    module.PREPROCESSING_DESCRIPTION = "identical Raw RGB aspect-preserving resize-pad for real and generated"


def patch_a3(module) -> None:
    module.MMOTU_CLASS_ORDER = CLASSES
    module.NUM_CLASSES = len(CLASSES)
    module.PROMPTS = PROMPTS
    module.preprocess_ultrasound_pil = preprocess_endoscopy_pil
    module.preprocess_ultrasound_mask = preprocess_endoscopy_prior
    module.STRUCTURE_SUPERVISION_DESCRIPTION = (
        "six pathology classes with train-image-derived Level-3 mucosal-gradient soft priors; no pixel mask"
    )


def patch_common(module) -> None:
    module.preprocess_ultrasound_pil = preprocess_endoscopy_pil
    module.preprocess_ultrasound_mask = preprocess_endoscopy_prior


def patch_a4_controller(module) -> None:
    module.MMOTU_CLASS_ORDER = CLASSES
    module.MMOTU_CLASS_SLUGS = SLUGS
    module.PROMPTS = PROMPTS
    module.NEGATIVE_PROMPT = NEGATIVE
    module.preprocess_ultrasound_pil = preprocess_endoscopy_pil
    module.preprocess_ultrasound_mask = preprocess_endoscopy_prior


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_hyperkvasir_pathology6_stage.py STAGE [stage arguments]")
    stage = sys.argv[1]
    sys.argv = [sys.argv[0], *sys.argv[2:]]

    if stage == "lora":
        import training.train_mmotu_lora as module
        patch_lora(module)
        module.main(); return
    if stage == "classifier":
        import training.run_mmotu_convnext as module
        patch_classifier(module)
        module.main(); return
    if stage == "generate":
        import training.train_mmotu_lora as lora
        patch_lora(lora)
        import training.generate_mmotu_balanced as module
        patch_generation(module); module.main(); return
    if stage == "quality":
        import training.evaluate_mmotu_quality as module
        patch_quality(module); module.main(); return
    if stage == "a3":
        import training.train_mmotu_lora as lora
        import training.run_mmotu_convnext as classifier
        patch_lora(lora); patch_classifier(classifier)
        import training.train_mmotu_a3_adversarial as module
        patch_a3(module); module.main(); return

    import training.run_mmotu_convnext as classifier
    patch_classifier(classifier)
    import training.mmotu_a4_common as common
    patch_common(common)
    if stage == "a4_diagnostic":
        import training.diagnose_mmotu_a4_coverage as module
        module.MMOTU_CLASS_ORDER = CLASSES
        module.main(); return
    if stage == "a4_controller":
        import training.train_mmotu_lora as lora
        patch_lora(lora)
        import training.run_mmotu_a4_adaptive as module
        patch_a4_controller(module); module.main(); return
    if stage == "ratio_build":
        import training.build_hyperkvasir_pathology6_ratio_manifests as module
        module.main(); return
    raise SystemExit(f"unknown stage: {stage}")


if __name__ == "__main__":
    main()
