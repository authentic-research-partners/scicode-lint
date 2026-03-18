import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


INPUT_SIZE = 640
BATCH_SIZE = 8


class DetectionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(8),
        )
        self.head = nn.Linear(128 * 8 * 8, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features.flatten(1))


class ObjectDetectionAPI:
    def __init__(self, model_path, device="cuda"):
        self.device = torch.device(device)
        torch.backends.cudnn.benchmark = True
        self.model = DetectionModel(num_classes=80)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _prepare_tensor(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image).to(self.device)

    def predict_single(self, image: Image.Image) -> torch.Tensor:
        tensor = self._prepare_tensor(image).unsqueeze(0)
        with torch.no_grad():
            return self.model(tensor).cpu()

    def predict_batch(self, images: list) -> torch.Tensor:
        tensors = torch.stack([self._prepare_tensor(img) for img in images])
        with torch.no_grad():
            return self.model(tensors).cpu()

    def serve(self, request_queue):
        batch_buffer = []
        results = []
        with torch.no_grad():
            for request in request_queue:
                image = Image.open(request["path"])
                batch_buffer.append(self._prepare_tensor(image))
                if len(batch_buffer) == BATCH_SIZE:
                    batch = torch.stack(batch_buffer)
                    predictions = self.model(batch)
                    results.extend(predictions.cpu().tolist())
                    batch_buffer = []
            if batch_buffer:
                batch = torch.stack(batch_buffer)
                padding = torch.zeros(
                    BATCH_SIZE - len(batch_buffer), 3, INPUT_SIZE, INPUT_SIZE,
                    device=self.device,
                )
                padded_batch = torch.cat([batch, padding])
                predictions = self.model(padded_batch)
                results.extend(predictions[:len(batch_buffer)].cpu().tolist())
        return results
