import torch
import torch.nn as nn
from torchvision import transforms
from diffusers import AutoencoderKL
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from constant import IMPLEMENTED_MODEL_ID


class VAE(nn.Module):
    def __init__(self, vae_from="sd15", norm_from_01: bool = True):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(IMPLEMENTED_MODEL_ID[vae_from]["model_id"], subfolder="vae")
        self.norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) if norm_from_01 else nn.Identity()

    def forward(self, x: torch.FloatTensor) -> DiagonalGaussianDistribution:
        posterior = self.vae.encode(self.norm(x)).latent_dist
        return posterior

    @staticmethod
    def sample(posterior: DiagonalGaussianDistribution,
               num_samples=1,
               random_seed=3407) -> torch.Tensor:
        samples = []
        for i in range(num_samples):
            generator = torch.manual_seed(random_seed)
            samples.append(posterior.sample(generator=generator))
            random_seed += 1
        return torch.cat(samples, dim=0)

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor, random_seed=3407):
        return torch.clamp(self.vae(x, generator=torch.manual_seed(random_seed)).sample, 0, 1)
