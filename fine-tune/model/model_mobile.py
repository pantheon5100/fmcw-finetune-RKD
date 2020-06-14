import torch
import torch.nn as nn
from torchvision import models


class Mobile(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.mobile = models.mobilenet_v2(pretrained=pretrained)
        self.features = self.mobile.features
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.mobile.last_channel, 6),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)

        return x


if __name__ == "__main__":
    model = Mobile()
    input_ = torch.randn([4, 3, 224, 224])
    out_ = model(input_)
    print(out_)
