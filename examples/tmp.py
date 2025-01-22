import torch
from torch import nn

class MixedCosineL1Loss(nn.Module):
    def __init__(self, alpha, beta):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.cos_criterion = nn.CosineEmbeddingLoss()
        self.smooth_l1_criterion = nn.SmoothL1Loss()

    def forward(self, y1: torch.Tensor, y2: torch.Tensor):
        y1 = y1.reshape(y1.shape[0], -1)
        y2 = y2.reshape(y2.shape[0], -1)

        target = torch.ones(y1.shape[0], device=y1.device)

        return self.alpha * self.cos_criterion(y1, y2, target) * self.beta * self.smooth_l1_criterion(y1, y2)
    
y1 = torch.randn((16,2,3))
y2 = torch.randn((16,2,3))
criterion = MixedCosineL1Loss(0.9, 0.1)
print(criterion(y1, y2))
