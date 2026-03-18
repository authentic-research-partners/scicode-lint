import torch
import torch.nn as nn


def create_model(input_dim):
    model = nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 10),
    )
    return model


def train(model, data, labels, criterion):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(100):
        for i in range(0, len(data), 32):
            batch = data[i : i + 32]
            batch_labels = labels[i : i + 32]
            loss = criterion(model(batch), batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
