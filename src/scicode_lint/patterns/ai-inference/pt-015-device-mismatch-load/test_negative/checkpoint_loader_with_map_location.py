import torch


class CheckpointLoader:
    def __init__(self, checkpoint_dir, device="cuda"):
        self.checkpoint_dir = checkpoint_dir
        self.device = torch.device(device)

    def load_latest(self, model):
        checkpoint_path = f"{self.checkpoint_dir}/latest.pt"
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state"])
        model.to(self.device)
        return model, checkpoint.get("epoch", 0)


def load_checkpoint_to_device(path, model, target_device):
    device = torch.device(target_device)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def resume_training(model, optimizer, checkpoint_path, device="cuda"):
    target_device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=target_device)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    model.to(target_device)
    return checkpoint["epoch"], checkpoint.get("best_loss", float("inf"))
