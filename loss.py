import torch
from torch import nn
import clip
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip


class CLIPLoss(nn.Module):
    """
    Directional CLIP Loss.

    Measures whether the change between the frozen and styled image
    follows the same direction as the change between the source and
    target text prompts in CLIP feature space.
    """

    def __init__(self, stylegan_size=1024, device="cuda"):
        super().__init__()

        self.device = device

        self.model, _ = clip.load("ViT-B/32", device=device)

        # Convert StyleGAN output (1024x1024) to CLIP input size
        self.upsample = nn.Upsample(scale_factor=7)
        self.avg_pool = nn.AvgPool2d(kernel_size=stylegan_size // 32)

    def encode_image(self, image):
        """
        Encode image with CLIP.
        """

        image = self.avg_pool(self.upsample(image))

        features = self.model.encode_image(image)
        features = features / features.norm(dim=-1, keepdim=True)

        return features

    def encode_text(self, text):
        """
        Encode text with CLIP.
        """

        tokens = clip.tokenize(text).to(self.device)

        features = self.model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)

        return features

    def forward(
        self,
        generated_image,
        frozen_image,
        source_text,
        target_text,
    ):
        """
        Args:
            generated_image : images from generated generator
            frozen_image : frozen image
            source_text : source prompt
            target_text : target prompt
        """

        image_features_frozen = self.encode_image(generated_image)
        image_features_styled = self.encode_image(frozen_image)

        text_features_source = self.encode_text(source_text)
        text_features_target = self.encode_text(target_text)

        # Direction vectors

        image_direction = image_features_styled - image_features_frozen
        text_direction = text_features_target - text_features_source

        image_direction = image_direction / (
            image_direction.norm(dim=-1, keepdim=True) + 1e-8
        )

        text_direction = text_direction / (
            text_direction.norm(dim=-1, keepdim=True) + 1e-8
        )

        similarity = F.cosine_similarity(
            image_direction,
            text_direction,
            dim=-1,
        ).clamp(-1, 1)

        loss = 1 - similarity.mean()

        return loss