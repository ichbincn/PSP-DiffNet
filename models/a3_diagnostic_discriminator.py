from __future__ import annotations

from typing import Dict, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_tiny


class A3DiagnosticDiscriminator(nn.Module):
    """Leakage-safe three-head discriminator for A3.

    The diagnostic class head receives image features only. Labels are used by
    the class-conditional realism projection and by losses outside this model;
    they are never an input to the class-consistency prediction.
    """

    def __init__(self, num_classes: int = 5, feature_dim: int = 768, mid_dim: int = 384):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = convnext_tiny(weights=None)
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)

        self.realism_label_embedding = nn.Embedding(num_classes, feature_dim)
        self.realism_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )
        self.structure_head = nn.Sequential(
            nn.Linear(mid_dim * 4 + 2, 256),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def load_medical_teacher(self, state_dict: Dict[str, torch.Tensor]) -> None:
        missing, unexpected = self.backbone.load_state_dict(state_dict, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"Teacher state mismatch: missing={missing}, unexpected={unexpected}")

    def encode(self, images: torch.Tensor):
        x = images
        mid = None
        for index, layer in enumerate(self.backbone.features):
            x = layer(x)
            if index == 5:
                mid = x
        if mid is None:
            raise RuntimeError("ConvNeXt features[5] was not reached")
        pooled = self.backbone.avgpool(x)
        normalized = self.backbone.classifier[0](pooled)
        final_feature = torch.flatten(normalized, 1)
        class_logits = self.backbone.classifier[2](final_feature)
        return mid, final_feature, class_logits

    def class_logits(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode(images)[2]

    def realism_from_features(self, final_feature: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        image_unit = F.normalize(final_feature, dim=1)
        label_unit = F.normalize(self.realism_label_embedding(labels), dim=1)
        projection = (image_unit * label_unit).sum(dim=1, keepdim=True)
        return self.realism_head(final_feature) + projection

    def structure_from_features(self, mid: torch.Tensor, prior_maps: torch.Tensor) -> torch.Tensor:
        prior = F.interpolate(prior_maps, size=mid.shape[-2:], mode="bilinear", align_corners=False).clamp(0, 1)
        inv_prior = 1.0 - prior
        denom_hi = prior.sum(dim=(2, 3)).clamp_min(1e-6)
        denom_lo = inv_prior.sum(dim=(2, 3)).clamp_min(1e-6)
        high = (mid * prior).sum(dim=(2, 3)) / denom_hi
        low = (mid * inv_prior).sum(dim=(2, 3)) / denom_lo
        interaction = (mid * (2.0 * prior - 1.0)).mean(dim=(2, 3))
        prior_mean = prior.mean(dim=(2, 3))
        prior_std = prior.flatten(2).std(dim=2, unbiased=False)
        feature = torch.cat([high, low, high - low, interaction, prior_mean, prior_std], dim=1)
        return self.structure_head(feature)

    def forward(self, images: torch.Tensor, labels: torch.Tensor, prior_maps: torch.Tensor):
        mid, final_feature, class_logits = self.encode(images)
        return {
            "class_logits": class_logits,
            "realism_logits": self.realism_from_features(final_feature, labels),
            "structure_logits": self.structure_from_features(mid, prior_maps),
        }

    def freeze_all(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def set_class_warmup_trainable(self) -> None:
        self.freeze_all()
        for parameter in self.backbone.classifier.parameters():
            parameter.requires_grad_(True)

    def set_aux_heads_trainable(self) -> None:
        self.freeze_all()
        for module in (self.realism_label_embedding, self.realism_head, self.structure_head):
            for parameter in module.parameters():
                parameter.requires_grad_(True)

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def describe(self) -> Dict[str, object]:
        return {
            "name": "A3DiagnosticDiscriminator",
            "encoder": "scratch ConvNeXt-Tiny initialized from real-train medical checkpoint",
            "heads": ["class_consistency", "class_conditional_realism", "spatial_structure_consistency"],
            "class_label_leakage": False,
            "class_head_input": "image feature only",
            "realism_conditioning": "projection with target class embedding",
            "structure_input": "14x14 image feature and weak localization prior",
            "parameters_total": sum(parameter.numel() for parameter in self.parameters()),
            "parameters_trainable_current": sum(
                parameter.numel() for parameter in self.parameters() if parameter.requires_grad
            ),
        }
