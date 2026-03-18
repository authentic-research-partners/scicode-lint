import torch
import torch.nn as nn
import torch.nn.functional as F


class AnomalyDetectorCNN(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.head(x)


def screen_patients(detector, scan_tensors, sensitivity_threshold=0.3):
    detector.eval()
    results = {}
    with torch.no_grad():
        raw_scores = detector(scan_tensors).squeeze(-1)
        probabilities = torch.sigmoid(raw_scores)
        flagged_mask = probabilities > sensitivity_threshold
        results["flagged_indices"] = torch.where(flagged_mask)[0]
        results["risk_scores"] = probabilities
    return results


def compute_screening_metrics(detector, loader, operating_point=0.5):
    detector.eval()
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    with torch.no_grad():
        for scans, labels in loader:
            raw = detector(scans).squeeze(-1)
            probs = torch.sigmoid(raw)
            preds = (probs >= operating_point).long()
            tp += ((preds == 1) & (labels == 1)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()
            tn += ((preds == 0) & (labels == 0)).sum().item()
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return {"sensitivity": sensitivity, "specificity": specificity}
