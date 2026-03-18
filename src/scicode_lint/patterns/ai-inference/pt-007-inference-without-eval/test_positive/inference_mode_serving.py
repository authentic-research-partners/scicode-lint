import torch
import torch.nn as nn


class ImageEncoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.classifier(x)


def serve_prediction(model, image_tensor):
    with torch.inference_mode():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=-1)
        top_k = torch.topk(probabilities, k=5, dim=-1)
    return top_k.indices.tolist(), top_k.values.tolist()


def run_batch_inference(model, dataloader, device):
    all_predictions = []
    with torch.inference_mode():
        for batch in dataloader:
            batch = batch.to(device)
            outputs = model(batch)
            preds = outputs.argmax(dim=1)
            all_predictions.extend(preds.cpu().tolist())
    return all_predictions
