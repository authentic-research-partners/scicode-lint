import torch
import torch.nn as nn

LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 100
HIDDEN_DIM = 256
DROPOUT = 0.3
NUM_LAYERS = 3


def create_model(input_dim: int, output_dim: int) -> nn.Module:
    layers = []
    dim = input_dim
    for _ in range(NUM_LAYERS):
        layers.extend([nn.Linear(dim, HIDDEN_DIM), nn.ReLU(), nn.Dropout(DROPOUT)])
        dim = HIDDEN_DIM
    layers.append(nn.Linear(dim, output_dim))
    return nn.Sequential(*layers)


def train(model, train_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(EPOCHS):
        for batch in train_loader:
            loss = nn.functional.cross_entropy(model(batch[0]), batch[1])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
