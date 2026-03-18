import pickle
from pathlib import Path


class CheckpointManager:

    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir

    def save_checkpoint(self, model, optimizer, epoch: int, loss: float) -> Path:
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss": loss,
        }
        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pkl"
        with open(path, "wb") as f:
            pickle.dump(checkpoint, f)
        return path

    def save_best_model(self, model, metrics: dict) -> Path:
        path = self.checkpoint_dir / "best_model.pkl"
        with open(path, "wb") as f:
            pickle.dump({"model": model, "metrics": metrics}, f)
        return path
