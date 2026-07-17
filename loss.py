import torch
from torch import nn
import clip
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from utils import *

class CLIPLoss(nn.Module):
    """
    Directional CLIP Loss из статьи StyleGAN-NADA.
    """

    def __init__(self, stylegan_size=1024, device="cuda"):
        super().__init__()

        self.device = device

        self.model, _ = clip.load("ViT-B/32", device=device)

        self.upsample = nn.Upsample(scale_factor=7)
        self.avg_pool = nn.AvgPool2d(kernel_size=stylegan_size // 32)

    def forward(
        self,
        generated_image,
        frozen_image,
        source_text,
        target_text,
    ):
        """
        Args:
            generated_image : изображение обучаемого генератора
            frozen_image    : изображение исходного (замороженного) генератора
            source_text     : исходный промпт
            target_text     : целевой промпт
        """

        generated_image = self.avg_pool(
            self.upsample(generated_image)
        )

        frozen_image = self.avg_pool(
            self.upsample(frozen_image)
        )

        source_tokens = clip.tokenize(source_text).to(self.device)
        target_tokens = clip.tokenize(target_text).to(self.device)

        generated_features = self.model.encode_image(generated_image)
        frozen_features = self.model.encode_image(frozen_image)

        source_features = self.model.encode_text(source_tokens)
        target_features = self.model.encode_text(target_tokens)

        generated_features = generated_features / generated_features.norm(dim=-1, keepdim=True)
        frozen_features = frozen_features / frozen_features.norm(dim=-1, keepdim=True)

        source_features = source_features / source_features.norm(dim=-1, keepdim=True)
        target_features = target_features / target_features.norm(dim=-1, keepdim=True)

        image_direction = generated_features - frozen_features
        text_direction = target_features - source_features

        image_direction = image_direction / (
            image_direction.norm(dim=-1, keepdim=True) + 1e-8
        )

        text_direction = text_direction / (
            text_direction.norm(dim=-1, keepdim=True) + 1e-8
        )

        loss = 1 - torch.cosine_similarity(
            image_direction,
            text_direction,
            dim=-1,
        ).mean()

        return loss
    