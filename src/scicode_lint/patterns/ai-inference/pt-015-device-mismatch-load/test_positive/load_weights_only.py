import torch
import torch.nn as nn


class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x).mean(dim=1)
        return self.fc(x)


def load_classifier(model_path, vocab_size=30000, embed_dim=128, num_classes=4):
    model = TextClassifier(vocab_size, embed_dim, num_classes)
    state_dict = torch.load(model_path, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_finetuned_head(base_model, head_path):
    head_weights = torch.load(head_path, weights_only=True)
    base_model.fc.load_state_dict(head_weights)
    return base_model
