import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch
from utils import NUM_PTS

class ConvBlock(nn.Module):
    def __init__(self, inp, oup, k, s, p, dw=False, linear=False):
        super(ConvBlock, self).__init__()
        self.linear = linear
        if dw:
            self.conv = nn.Conv2d(inp, oup, k, s, p, groups=inp, bias=False)
        else:
            self.conv = nn.Conv2d(inp, oup, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        if not linear:
            self.prelu = nn.PReLU(oup)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.linear:
            return x
        else:
            return self.prelu(x)


class SEModule(nn.Module):
    '''Squeeze and Excitation Module'''
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return input * x


class RESNEXT_steroid(nn.Module):
    def __init__(self):
        super(RESNEXT_steroid, self).__init__()
        model = models.resnext50_32x4d()
        model.fc = nn.Linear(model.fc.in_features, 2 * NUM_PTS, bias=True)
        checkpoint = torch.load("./runs/baseline_full4_best.pth", map_location='cpu')
        model.load_state_dict(checkpoint, strict=True)
        self.base_net = nn.Sequential(*list(model.children())[:-2])
        out_size = model.fc.in_features
        self.linear7 = ConvBlock(out_size, out_size, (4, 4), 1, 0, dw=True, linear=True) #(7x7)
        self.linear1 = ConvBlock(out_size, out_size, 1, 1, 0, linear=True)
        self.attention = SEModule(out_size, 8)
        self.fc = nn.Linear(out_size, 2 * NUM_PTS, bias=True)


    def forward(self, x):
        x = self.base_net(x)
        x = self.attention(x)
        x = self.linear7(x)
        x = self.linear1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x