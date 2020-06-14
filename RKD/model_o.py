import torch
import torch.nn as nn
from torchvision import models

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

    def forward(self, input, get_ha=False):
        x = self.conv1(input)
        x = self.relu(self.bn1(x))
        x = self.maxpool1(x)
        feature1 = self.layer1(x)             # 1 / 4
        feature2 = self.layer2(feature1)      # 1 / 8
        feature3 = self.layer3(feature2)      # 1 / 16
        feature4 = self.layer4(feature3)      # 1 / 32
        if get_ha:
            return feature1, feature2, feature3, feature4
        return feature4


class ResFT(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.feature = resnet18(pretrained=pretrained)
        self.maxpool = nn.MaxPool2d((8, 8))
        self.fc1 = torch.nn.Linear(512, 512)
        self.relu1 = torch.nn.ReLU(True)
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.fc2 = torch.nn.Linear(512, 6)
        # global average pooling to build tail


    def forward(self, inputs, get_ha=False):
        inputs = torch.nn.functional.interpolate(inputs, size=(256, 256), mode='bilinear')
        if get_ha:
            b1, b2, b3, b4 = self.feature(inputs, get_ha=get_ha)
            tail = self.maxpool(b4)
        else:
            tail = self.feature(inputs)
            tail = self.maxpool(tail)
        X = self.fc1(tail.view((-1, 512)))
        X = self.relu1(X)
        X = self.bn1(X)
        X = self.fc2(X)

        if get_ha:
            return  b1, b2, b3, b4, tail, X

        return X


if __name__ == '__main__':
    from thop import profile
    model = ResFT()
    input_ = torch.randn([2,3,256,256])
    # param = profile(model, inputs=(input_,))
    from torchviz import make_dot
    y = model(input_)
    g=make_dot(y)
    g.render('tea')
    # print(param)
    pass
