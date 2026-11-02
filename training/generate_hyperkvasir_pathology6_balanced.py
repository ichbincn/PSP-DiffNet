"""Generate the fixed HyperKvasir pathology6 Raw A0 pool with shared SD code.

This adapter deliberately exposes only ``a0``: ordinary comparisons use the
Raw image path and no mask, contour, weak prior or lesion ROI.  The supplied
Stable-Diffusion/LoRA img2img implementation remains unchanged.
"""
from __future__ import annotations

import pathlib
import sys
from dataclasses import replace

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import training.generate_mmotu_balanced as generator  # noqa: E402
from data.hyperkvasir_pathology import (  # noqa: E402
    HYPERKVASIR_PATHOLOGY6_CLASS_ORDER,
    HYPERKVASIR_PATHOLOGY6_CLASS_SLUGS,
    HYPERKVASIR_PATHOLOGY6_PROMPTS,
)
from utils.endoscopy import preprocess_endoscopy_pil  # noqa: E402


generator.MMOTU_CLASS_ORDER = list(HYPERKVASIR_PATHOLOGY6_CLASS_ORDER)
generator.MMOTU_CLASS_SLUGS = list(HYPERKVASIR_PATHOLOGY6_CLASS_SLUGS)
generator.PROMPTS = dict(HYPERKVASIR_PATHOLOGY6_PROMPTS)
generator.preprocess_ultrasound_pil = preprocess_endoscopy_pil


def _pathology_tasks(frame, seed):
    """A mask-free equivalent of the shared task builder."""
    rng = np.random.default_rng(seed)
    target = int(frame.label.value_counts().max())
    tasks = []
    for label, (class_name, slug) in enumerate(
        zip(generator.MMOTU_CLASS_ORDER, generator.MMOTU_CLASS_SLUGS)
    ):
        class_frame = frame[frame.label == label].reset_index(drop=True)
        need = target - len(class_frame)
        if need <= 0:
            continue
        sources = class_frame.iloc[rng.permutation(len(class_frame))].to_dict("records")
        for index in range(need):
            source = sources[index % len(sources)]
            tasks.append(
                generator.GenerationTask(
                    output_index=index,
                    label=label,
                    label_name=class_name,
                    label_slug=slug,
                    source_image_id=str(source["image_id"]),
                    source_path=str(pathlib.Path(source["image_path"]).resolve()),
                    prior_path="",
                    seed=seed + label * 100_000 + index,
                )
            )
    return tasks


generator.build_tasks = _pathology_tasks
generator.PRIOR_SOURCE_DESCRIPTION = "none; Raw image-to-image only"


if __name__ == "__main__":
    # The shared parser/runner enforces method=a0 and the frozen train-only
    # task/quota bookkeeping.  The wrapper is intentionally not a new model.
    generator.main()
