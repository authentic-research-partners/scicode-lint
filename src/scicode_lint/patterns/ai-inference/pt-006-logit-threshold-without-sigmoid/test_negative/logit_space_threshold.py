import math

import torch
import torch.nn as nn


class FraudDetector(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


def detect_fraud(model, transactions, prob_threshold=0.3):
    model.eval()
    logit_threshold = math.log(prob_threshold / (1 - prob_threshold))
    with torch.no_grad():
        logits = model(transactions).squeeze(-1)
        flagged = logits > logit_threshold
    return flagged
