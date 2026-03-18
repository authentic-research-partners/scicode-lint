import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        return self.fc(x)


def export_model(model, output_path):
    model.eval()
    dummy_input = torch.randn(1, 512)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
    )
