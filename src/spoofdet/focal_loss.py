import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftTargetFocalLoss(nn.Module):
    def __init__(self, alpha=[0.25, 0.75], gamma=2.0, reduction="mean"):
        """
        alpha: list of weights for each class (e.g., [weight_for_class_0, weight_for_class_1])
        """
        super(SoftTargetFocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha) if alpha is not None else None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: (N, C) - Logits (Raw scores)
        # targets: (N, C) - Soft Labels from MixUp (e.g., [0.2, 0.8])

        # 1. Convert Logits to Probabilities
        probs = F.softmax(inputs, dim=1)

        # 2. Calculate the "Focal Term" (1 - p)^gamma
        # Note: We compute log_probs for numerical stability
        log_probs = F.log_softmax(inputs, dim=1)
        focal_term = (1 - probs) ** self.gamma

        # 3. Apply Alpha Weighting (Element-wise)
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            # Expand alpha to batch size
            alpha_t = self.alpha.expand(inputs.size(0), -1)
            loss_components = -1 * targets * alpha_t * focal_term * log_probs
        else:
            loss_components = -1 * targets * focal_term * log_probs

        # 4. Sum across classes, then reduce across batch
        # Summing over dim=1 (classes) captures the mix of both classes
        loss = loss_components.sum(dim=1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
