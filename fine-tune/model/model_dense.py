import torch
import torch.nn as nn
from torchvision import models
from tensorboardX import SummaryWriter
import torch.nn.functional as F


class DenseNet(torch.nn.Module):
    def __init__(self, pretrained=True, arc='121'):
        super().__init__()
        if arc == '121':
            self.features = models.densenet121(pretrained=pretrained)
        elif arc=='161':
            self.features = models.densenet161(pretrained=pretrained)
        else:
            self.features = models.densenet121(pretrained=pretrained)
        self.feature = self.features.features

    def forward(self, input):
        x = self.feature(input)
        out = F.relu(x, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)

        return out


class Dense(torch.nn.Module):
    def __init__(self, pretrained=True, arc='121'):
        super().__init__()
        self.dense = DenseNet(pretrained=pretrained, arc=arc)
        if arc == '121':
            self.fc = torch.nn.Linear(1024, 6)
        elif arc == '161':
            self.fc = torch.nn.Linear(2048, 6)

        else:
            self.fc = torch.nn.Linear(1024, 6)

    def forward(self, x):
        x = self.dense(x)
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

