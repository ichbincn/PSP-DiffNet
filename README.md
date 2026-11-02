# PSP-DiffNet

Core research code for diffusion-based augmentation of imbalanced medical image
classification datasets: APTOS2019 fundus images, MMOTU/OTU_2d ultrasound, and
six-class HyperKvasir pathology images.

The pipeline combines Stable Diffusion 1.5 medical-domain LoRA, structural
priors, online diagnostic-consistency feedback, and adaptive quality/coverage
control. Synthetic images are used only in the classifier training split;
validation and test splits contain real images.

## Contents

| Directory | Contents |
| --- | --- |
| `models/` | Three-head diagnostic discriminator |
| `training/` | LoRA training, generation, structural priors, diagnostic feedback, adaptive selection and downstream classifiers |
| `data/` | Dataset readers, class definitions, split utilities and imbalance handling; Python code only |
| `metrics/` | Classification metrics and FID/KID/improved precision-recall evaluation |
| `utils/` | Fundus, ultrasound and endoscopy preprocessing and shared helpers |

## Setup

Use a Python 3.10 environment. Training is designed for an NVIDIA GPU with CUDA;
the original experiments used PyTorch 2.0/CUDA 11.8. Install a compatible
PyTorch/torchvision pair for your machine, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

Obtain the datasets and pretrained Stable Diffusion 1.5 weights separately under
their respective terms. By default, the diffusion scripts expect a local Diffusers
model directory at `models/pretrained/stable-diffusion-v1-5`; use `--model-dir`
to select another location. Evaluation may also download torchvision and
pytorch-fid feature-extractor weights.

## Main entry points

Run commands from the repository root. Each entry point exposes its arguments
with `--help`.

| Component | Entry point |
| --- | --- |
| Fundus LoRA | `training/train_diffusion_lora.py` |
| Fundus generation | `training/generate_aptos_lora_sd025_balanced.py` |
| Fundus structural prior / guided generation | `training/prepare_aptos_structure_priors.py`, `training/generate_aptos_structure_guided.py` |
| Fundus diagnostic feedback | `training/train_a3_adversarial.py` |
| Fundus adaptive control | `training/diagnose_a4v2_medical_local.py`, `training/run_a4v2_diagnostic_controller.py` |
| Ultrasound ROI / mask prior | `training/prepare_mmotu_roi_mainline.py`, `training/prepare_mmotu_gt_priors.py` |
| Ultrasound LoRA / generation | `training/train_mmotu_lora.py`, `training/generate_mmotu_balanced.py` |
| Ultrasound diagnostic feedback | `training/train_mmotu_a3_adversarial.py` |
| Ultrasound adaptive control | `training/diagnose_mmotu_a4_coverage.py`, `training/run_mmotu_a4_adaptive.py` |
| HyperKvasir preparation / weak prior | `training/prepare_hyperkvasir_pathology6_input.py`, `training/prepare_hyperkvasir_pathology6_generation.py` |
| HyperKvasir shared pipeline | `training/run_hyperkvasir_pathology6_stage.py` |
| Downstream classification | `training/run_aptos_convnext_baselines.py`, `training/run_mmotu_convnext.py` |

Examples:

```bash
python -m training.train_diffusion_lora --help
python -m training.train_diffusion_lora \
  --model-dir models/pretrained/stable-diffusion-v1-5 \
  --train-split /path/to/aptos/train.csv \
  --output-dir outputs/aptos/lora

python -m training.run_hyperkvasir_pathology6_stage a3 --help
python -m training.run_mmotu_convnext --help
```

The HyperKvasir stage adapter supports `lora`, `generate`, `classifier`, `quality`,
`a3`, `a4_diagnostic`, `a4_controller` and `ratio_build`. For the paper's classifier
protocol, use `--initialization imagenet` for MMOTU and `--initialization scratch`
for HyperKvasir. The APTOS baseline classifier uses scratch initialization.

## Inputs and workflow

Prepare real train/validation/test CSVs with valid local image paths before
training. Common columns are `image_id`, `image_path`, `label` and `label_name`;
ultrasound preparation also requires `mask_path`. Follow the class orders in
`data/aptos_dataset.py`, `data/mmotu_dataset.py` and `data/hyperkvasir_pathology.py`.
APTOS generation roots use `splits/{train,val,test}.csv`; MMOTU stages use
`{train,validation,test}.csv`. Inspect the selected entry point for its exact
manifest and prior-index inputs.

The usual order is real-data preparation and real-only classifier/teacher
training, LoRA adaptation, prior-guided generation, diagnostic-feedback training,
adaptive pool selection, then downstream evaluation. Later stages need artifacts
produced by earlier stages, including LoRA adapters, discriminator/teacher
checkpoints, prior indexes and generated-image manifests. 