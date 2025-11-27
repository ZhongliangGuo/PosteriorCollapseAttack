import torch
import torch.nn as nn
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution


class PosteriorCollapseLoss(nn.Module):
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        assert reduction in ['mean', 'none', 'sum']
        self.reduction = reduction

    def forward(self, posterior: DiagonalGaussianDistribution, v: [float, torch.Tensor] = 1e-8,
                reverse_direct=False) -> torch.Tensor:
        B = posterior.mean.size(0)
        loss = 0.5 * torch.sum(1 + posterior.logvar.view(B, -1) - (1 / v) * (
                    posterior.mean.view(B, -1) ** 2 + posterior.std.view(B, -1) ** 2), dim=-1)
        if reverse_direct:
            loss = -loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise NotImplemented(f'{self.reduction} has not been implemented.')

    def _get_name(self):
        return "ReverseKL"
