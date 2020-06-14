import torch
import torch.nn as nn
from torchvision import models
from tensorboardX import SummaryWriter


class resnet18(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.features = models.resnet18(pretrained=pretrained)
        self.conv1 = self.features.conv1
        self.bn1 = self.features.bn1
        self.relu = self.features.relu
        self.maxpool1 = self.features.maxpool
        self.layer1 = self.features.layer1
        self.layer2 = self.features.layer2
        self.layer3 = self.features.layer3
        self.layer4 = self.features.layer4
        # self.fc = torch.nn.Linear(512, 6)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(self.bn1(x))
        x = self.maxpool1(x)
        feature1 = self.layer1(x)             # 1 / 4
        feature2 = self.layer2(feature1)      # 1 / 8
        feature3 = self.layer3(feature2)      # 1 / 16
        feature4 = self.layer4(feature3)      # 1 / 32
        # global average pooling to build tail
        # tail = nn.MaxPool2d((8, 8))(feature4)
        x = self.avgpool(feature4)
        x = torch.flatten(x, 1)

        return x


class Shollow(torch.nn.Module):
    def __init__(self, out_d=6, pretrained=True):
        super().__init__()
        self.feature = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, stride=1),
                                     nn.MaxPool2d(kernel_size=2, stride=2),
                                     nn.Conv2d(16, 32, kernel_size=3, stride=1),
                                     nn.MaxPool2d(kernel_size=2, stride=2),
                                     nn.Conv2d(32, 64, kernel_size=3, stride=1),
                                     nn.MaxPool2d(kernel_size=2, stride=2),
                                     nn.Conv2d(64, 128, kernel_size=3, stride=1),
                                     nn.MaxPool2d(kernel_size=2, stride=2),
                                     nn.AdaptiveAvgPool2d((1, 1))
                                     )
        self.fc = torch.nn.Linear(128, out_d)

    def forward(self, x):
        x = self.feature(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    model = Shollow()
    input_ = torch.randn([4, 3, 224, 224])
    out_ = model(input_)
    print(out_)
