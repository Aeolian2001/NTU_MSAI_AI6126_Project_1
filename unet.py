import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchvision import models


# 编码块
class UNetEnc(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, 3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        ]
        if dropout:
            layers += [nn.Dropout(0.5)]
        layers += [nn.MaxPool2d(2, stride=2, ceil_mode=True)]

        self.down = nn.Sequential(*layers)

    def forward(self, x):
        return self.down(x)


# 解码块
class UNetDec(nn.Module):
    def __init__(self, in_channels, features, out_channels):
        super().__init__()

        self.up = nn.Sequential(
            nn.Conv2d(in_channels, features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(features),
            nn.ConvTranspose2d(features, out_channels, 2, stride=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)


# U-Net
class UNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.enc1 = UNetEnc(3, 16)
        self.enc2 = UNetEnc(16, 32)
        self.enc3 = UNetEnc(32, 64)
        self.enc4 = UNetEnc(64, 128, dropout=True)

        # Center block
        self.center = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Dropout(),
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ReLU(inplace=True),
        )

        # 解码块
        self.dec4 = UNetDec(256, 128, 64)
        self.dec3 = UNetDec(128, 64, 32)
        self.dec2 = UNetDec(64, 32, 16)
        self.dec1 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
        )

        self.final = nn.Conv2d(16, num_classes, 1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        center = self.center(enc4)

        dec4 = self.dec4(
            torch.cat([center, F.interpolate(enc4, center.size()[2:], mode='bilinear', align_corners=False)], 1))
        dec3 = self.dec3(
            torch.cat([dec4, F.interpolate(enc3, dec4.size()[2:], mode='bilinear', align_corners=False)], 1))
        dec2 = self.dec2(
            torch.cat([dec3, F.interpolate(enc2, dec3.size()[2:], mode='bilinear', align_corners=False)], 1))
        dec1 = self.dec1(
            torch.cat([dec2, F.interpolate(enc1, dec2.size()[2:], mode='bilinear', align_corners=False)], 1))

        return F.interpolate(self.final(dec1), size=x.size()[2:], mode='bilinear', align_corners=False)


# model = UNet(19)
# print(sum(p.numel() for p in model.parameters()))