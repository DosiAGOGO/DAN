import torch.nn as nn
import pytorch.mmdRe as mmd
import torch.nn.functional as F
import torch


class DANNet(nn.Module):

    def __init__(self, norm_layer=None):
        super(DANNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.layer1 = nn.Linear(6, 6)
        self.layer2 = nn.Linear(6, 36)
        self.layer3 = nn.Linear(36, 64)
        self.layer4_n = nn.Linear(64, 128)
        self.layer5_n = nn.Linear(128, 64)
        self.layer6_n = nn.Linear(64, 32)
        self.layer7_n = nn.Linear(32, 6)
        self.layer8_n = nn.Linear(6, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4_n(x))
        x = F.relu(self.layer5_n(x))
        x = F.relu(self.layer6_n(x))
        x = F.relu(self.layer7_n(x))
        x = F.relu(self.layer8_n(x))

        return x


class DANmain(nn.Module):

    def __init__(self):
        super(DANmain, self).__init__()
        self.sharedNet = DANNet(True)

    def forward(self, source, target):
        loss = 0
        source = self.sharedNet(source)
        if self.training == True:
            target = self.sharedNet(target)
            # loss += mmd.mmd_rbf_accelerate(source, target)
            loss += mmd.mmd_rbf_noaccelerate(source, target)

        return source, loss


def DAN(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DANmain()
    if pretrained:
        print("pretrained！")
        model.load_state_dict(torch.load("/media/zzg/29b7a8df-b813-4851-a749-f9db6d488d0d/zzg/Documents/DAN_raw_0-1w.pth"), strict=False)

    return model