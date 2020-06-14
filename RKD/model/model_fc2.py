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


class ResFT(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.resnet = resnet18(pretrained=pretrained)
        self.fc1 = torch.nn.Linear(512, 512)
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.relu1 = torch.nn.ReLU(True)
        self.fc2 = torch.nn.Linear(512, 6)

    def forward(self, x):
        x = self.resnet(x)
        X = self.fc1(x)
        X = self.relu1(X)
        X = self.bn1(X)
        x = self.fc2(X)

        return x

# if __name__ == '__main__':
#     model = not
#     print(model.state_dict())
#
#     # writer = SummaryWriter(comment='distribution-o')
#     # print(model.state_dict())
#     # writer.add_histogram('distribution centers3', model.state_dict()['fc2.bias'],0)
#     # writer.add_histogram('distribution centers3', model.state_dict()['fc2.bias'],1)
#     print(model.state_dict())
#     # writer.close()

