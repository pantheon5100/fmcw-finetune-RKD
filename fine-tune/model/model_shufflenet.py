import torch
import torch.nn as nn
from torchvision import models


class Shuffle(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.shuffle = models.shufflenet_v2_x0_5(pretrained=pretrained)
        # self.features = self.mobile.features
        self.conv1 = self.shuffle.conv1
        self.maxpool = self.shuffle.maxpool
        self.stage2 = self.shuffle.stage2
        self.stage3 = self.shuffle.stage3
        self.stage4 = self.shuffle.stage4
        self.conv5 = self.shuffle.conv5

        self.fc = nn.Linear(self.shuffle._stage_out_channels[-1], 6)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])  # globalpool
        x = self.fc(x)

        return x


if __name__ == "__main__":
    model = Mobile()
    input_ = torch.randn([4, 3, 224, 224])
    out_ = model(input_)
    print(out_)
