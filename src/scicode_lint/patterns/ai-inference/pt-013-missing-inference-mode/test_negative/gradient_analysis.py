import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_tensor, target_class):
        input_tensor = input_tensor.requires_grad_(True)
        output = self.model(input_tensor)
        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear")
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.squeeze().detach().cpu().numpy()


def compute_saliency_map(model, input_tensor, target_class):
    model.eval()
    input_tensor = input_tensor.clone().requires_grad_(True)
    output = model(input_tensor)
    score = output[0, target_class]
    score.backward()
    saliency = input_tensor.grad.data.abs().max(dim=1)[0]
    return saliency.cpu().numpy()
