from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F


class InitializeEfficientNetModel:
    """Class to initialize the model for training or inference"""

    def __init__(self, device: torch.device):
        self.device = device

    def get_model(self, with_weights: bool = False) -> torch.nn.Module:
        """getting the model for either training or inference"""

        if self.model_name == "efficientnet_v2_s":
            if with_weights:
                model = models.efficientnet_v2_s(
                    weights=models.EfficientNet_V2_S_Weights.DEFAULT
                )
            else:
                model = models.efficientnet_v2_s(weights=None)

            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, 2)

        else:
            raise ValueError(f"Model {self.model_name} not supported.")

        return model.to(self.device)

    def _freeze_stages(self, model: torch.nn.Module, num_unfrozen_stages: int):
        """Freezes the initial layers of the model based on num_unfrozen_stages"""

        # Total layers in EfficientNet v2 small backbone
        total_layers = 17  # Adjust based on actual architecture if needed

        # Calculate number of layers to freeze
        num_layers_to_freeze = total_layers - num_unfrozen_stages

        layer_count = 0
        for name, child in model.features.named_children():
            if layer_count < num_layers_to_freeze:
                for param in child.parameters():
                    param.requires_grad = False
            layer_count += 1

        print(
            f"Froze {num_layers_to_freeze} layers. Unfrozen layers: {num_unfrozen_stages}"
        )


def adaptive_batch_norm(model, data_loader, device, num_batches=100):
    """
    Resets and recalculates BatchNorm statistics on the target data.

    Args:
        model: The pretrained model (e.g., EfficientNetV2).
        data_loader: DataLoader for YOUR facial dataset (train or a subset).
        device: torch.device
        num_batches: Number of batches to use for recalibration.
    """
    import torch.nn as nn

    # 1. Set all BatchNorm layers to training mode
    # This is CRUCIAL: Enables calculation of running stats
    model.train()

    # 2. Reset ALL BatchNorm running statistics
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            module.reset_running_stats()
            # Optional: Reset momentum to use current batch more strongly
            module.momentum = 0.1  # Default is 0.1. Lower = more weight on new data.

    # 3. Forward pass on target data WITHOUT backpropagation
    with torch.no_grad():
        batches_processed = 0
        for images, _ in data_loader:  # We only need the images
            if batches_processed >= num_batches:
                break

            images = images.to(device)
            _ = model(images)  # Forward pass updates BN running stats

            batches_processed += 1
            if batches_processed % 20 == 0:
                print(f"  AdaBN: Processed {batches_processed} batches...")

    print(f"Adaptive BN complete. Calibrated on {batches_processed} batches.")
    return model
