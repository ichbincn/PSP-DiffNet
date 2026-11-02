"""Shared APTOS diffusion prompts retained by the frozen A0/A1/A4 generators.

The superseded plain-SD executable was removed during the 2026-08-15 cleanup.
Current code imports only these immutable prompt constants from its former module.
"""

PROMPTS = {
    "No_DR": "high quality color fundus photograph, healthy retina, no diabetic retinopathy, clinically plausible retinal anatomy",
    "Mild": "high quality color fundus photograph, mild diabetic retinopathy, clinically plausible retinal anatomy",
    "Moderate": "high quality color fundus photograph, moderate diabetic retinopathy, clinically plausible retinal anatomy",
    "Severe": "high quality color fundus photograph, severe diabetic retinopathy, clinically plausible retinal anatomy",
    "Proliferate_DR": "high quality color fundus photograph, proliferative diabetic retinopathy, clinically plausible retinal anatomy",
}

NEGATIVE_PROMPT = (
    "non-medical image, illustration, cartoon, text, watermark, distorted eye, "
    "duplicate vessels, unrealistic fundus, gray image, blank image, low quality, heavy artifact"
)
