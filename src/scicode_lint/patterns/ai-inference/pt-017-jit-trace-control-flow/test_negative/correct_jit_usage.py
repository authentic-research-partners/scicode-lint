import torch
import torch.nn as nn


class StaticModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def script_model_with_control_flow(model, example_input):
    model.eval()
    scripted_model = torch.jit.script(model)
    return scripted_model


def trace_static_model():
    model = StaticModel(64, 128, 10)
    model.eval()

    example_input = torch.randn(1, 64)
    traced = torch.jit.trace(model, example_input)
    return traced


def export_static_model(output_path):
    model = StaticModel(32, 64, 5)
    model.eval()

    scripted_model = torch.jit.script(model)
    scripted_model.save(output_path)
    return scripted_model
