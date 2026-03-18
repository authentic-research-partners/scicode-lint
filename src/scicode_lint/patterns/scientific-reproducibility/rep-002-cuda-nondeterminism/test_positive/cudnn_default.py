import torch
import torch.nn as nn


def train_cnn(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            inputs = inputs.cuda()
            targets = targets.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()


def setup_model():
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3),
        nn.ReLU(),
    ).cuda()
    return model
