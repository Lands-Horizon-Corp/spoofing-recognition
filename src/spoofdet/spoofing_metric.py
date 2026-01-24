import torch
from torchmetrics import Metric


class SpoofingMetric(Metric):
    """
    Computes APCER, BPCER, and ACER for Face Anti-Spoofing.

    Assumptions:
    - Target 1 = Live / Bona Fide
    - Target 0 = Spoof / Attack
    - Preds  = Probabilities (0.0 to 1.0) of being Live
    """

    # These state variables will accumulate data across batches
    full_state_update: bool = False

    def __init__(self, threshold: float = 0.5, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.threshold = threshold

        # We register the state to accumulate predictions and targets over batches
        # dist_reduce_fx="cat" ensures data is concatenated correctly in distributed training
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Add a batch of predictions and targets.
        """
        # Ensure preds are 1D (probabilities) and targets are 1D
        preds = preds.squeeze()
        target = target.squeeze()

        self.preds.append(preds)
        self.targets.append(target)

    def compute(self):
        """
        Compute the final APCER, BPCER, and ACER based on all accumulated data.
        Returns a dictionary containing the three metrics.
        """
        # Concatenate the list of tensors into a single tensor
        preds = torch.cat(self.preds)
        targets = torch.cat(self.targets)

        # 1. Separate Bona Fide (Live) and Attack (Spoof)
        # Target 1 = Live, Target 0 = Spoof
        is_bonafide = targets == 1
        is_attack = targets == 0

        bonafide_scores = preds[is_bonafide]
        attack_scores = preds[is_attack]

        # 2. Calculate APCER (Attack Presentation Classification Error Rate)
        # Proportion of Attack samples incorrectly classified as Live (Score > Threshold)
        if len(attack_scores) > 0:
            false_accepts = (attack_scores > self.threshold).float().sum()
            apcer = false_accepts / len(attack_scores)
        else:
            apcer = torch.tensor(0.0, device=self.device)

        # 3. Calculate BPCER (Bona Fide Presentation Classification Error Rate)
        # Proportion of Live samples incorrectly classified as Attack (Score <= Threshold)
        if len(bonafide_scores) > 0:
            false_rejects = (bonafide_scores <= self.threshold).float().sum()
            bpcer = false_rejects / len(bonafide_scores)
        else:
            bpcer = torch.tensor(0.0, device=self.device)

        # 4. Calculate ACER (Average Classification Error Rate)
        acer = (apcer + bpcer) / 2.0

        return {"APCER": apcer, "BPCER": bpcer, "ACER": acer}
