import torch
import torch.nn as nn


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 256)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def inference_server(model, request_queue):
    model.eval()
    model.cuda()

    torch.backends.cudnn.benchmark = True

    with torch.no_grad():
        while True:
            batch = request_queue.get()
            if batch is None:
                break

            images = batch["images"].cuda()
            embeddings = model(images)
            yield embeddings.cpu()


def preprocess_images(images_list):
    processed = []
    for images in images_list:
        normalized = (images.float() - 127.5) / 127.5
        processed.append(normalized)
    return processed
