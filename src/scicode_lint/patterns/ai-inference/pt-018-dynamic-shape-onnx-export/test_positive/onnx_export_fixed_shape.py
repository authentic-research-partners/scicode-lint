import torch
import torch.nn as nn


class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        _, hidden = self.rnn(embedded)
        return self.fc(hidden.squeeze(0))


def export_model(model, output_path):
    model.eval()
    dummy_ids = torch.randint(0, 5000, (1, 64))
    torch.onnx.export(
        model,
        dummy_ids,
        output_path,
        input_names=["token_ids"],
        output_names=["sentiment_scores"],
        verbose=True,
        opset_version=14,
    )


def run_export():
    model = SentimentRNN(vocab_size=5000, embed_dim=128, hidden_dim=256, num_classes=3)
    checkpoint = torch.load("sentiment_model.pth", weights_only=True)
    model.load_state_dict(checkpoint)
    export_model(model, "sentiment_model.onnx")
