import torch
import torch.nn as nn
from torchvision import models
from tensorboardX import SummaryWriter


class resnet18(torch.nn.Module):
    def __init__(self, pretrained=True, arc='18'):
        super().__init__()
        if arc == '18':
            self.features = models.resnet18(pretrained=pretrained)
        elif arc=='50':
            self.features = models.resnet50(pretrained=pretrained)
        elif arc=='101':
            self.features = models.resnet101(pretrained=pretrained)
        else:
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
    def __init__(self, pretrained=True, arc='18'):
        super().__init__()
        self.resnet = resnet18(pretrained=pretrained, arc=arc)
        if arc == '18':
            self.fc = torch.nn.Linear(512, 6)
        elif arc == '50':
            self.fc = torch.nn.Linear(2048, 6)
        elif arc == '101':
            self.fc = torch.nn.Linear(2048, 6)
        else:
            self.fc = torch.nn.Linear(512, 6)

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)

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

