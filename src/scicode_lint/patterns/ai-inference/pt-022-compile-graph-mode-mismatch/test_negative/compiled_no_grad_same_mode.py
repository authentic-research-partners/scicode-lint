import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def training_with_validation_loss(model, train_loader, val_loader, optimizer):
    compiled_model = torch.compile(model)

    compiled_model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        loss = compiled_model(batch).mean()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        val_loss = 0.0
        for batch in val_loader:
            val_loss += compiled_model(batch).mean().item()

    return val_loss / len(val_loader)
