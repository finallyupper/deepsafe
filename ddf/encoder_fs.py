import torch
import torch.nn as nn
from torchvision import models


def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


class ResNetUNet(nn.Module):
    def __init__(self, message_size):
        super().__init__()

        self.message_size = message_size
        self.base_model = models.resnet18(pretrained=False)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64 + message_size, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])  # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64 + message_size, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128 + message_size, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256 + message_size, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512 + message_size, 512, 1, 0)

        self.cbam2 = CBAM(channel=128)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, 3, 1)

    def forward(self, input, message):
        batch_size, _, h, w = input.shape
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original) #size=(N, 64, 256, 256)

        layer0 = self.layer0(input) # size=(N, 64, 128, 128)
        layer1 = self.layer1(layer0) # size=(N, 64, 64, 64)
        layer2 = self.layer2(layer1) # size=(N, 128, 32, 32)
        layer2 = self.cbam2(layer2) # size=(N, 128, 32, 32)
        layer3 = self.layer3(layer2) # size=(N, 256, 16, 16)
        layer4 = self.layer4(layer3) # size=(N, 512, 8, 8)
        layer4 = self.layer4_1x1(torch.cat(
            (layer4, message.expand(h // 32, w // 32, batch_size, self.message_size).permute(2, 3, 0, 1).contiguous()),
            dim=1))  # size=(N, 512, 8, 8)
        x = self.upsample(layer4) # size=(N, 512, 16, 16)

        layer3 = self.layer3_1x1(torch.cat( 
            (layer3, message.expand(h // 16, w // 16, batch_size, self.message_size).permute(2, 3, 0, 1).contiguous()),
            dim=1)) # size=(N, 256, 16, 16)
        x = torch.cat([x, layer3], dim=1) # size=(N, 512+256, 16, 16)
        x = self.conv_up3(x) # size=(N, 512, 16, 16)
        x = self.upsample(x) # size=(N, 512, 32, 32)

        layer2 = self.layer2_1x1(torch.cat(
            (layer2, message.expand(h // 8, w // 8, batch_size, self.message_size).permute(2, 3, 0, 1).contiguous()),
            dim=1))
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)
        x = self.upsample(x)  # size=(N, 256, 64, 64)

        layer1 = self.layer1_1x1(torch.cat(
            (layer1, message.expand(h // 4, w // 4, batch_size, self.message_size).permute(2, 3, 0, 1).contiguous()),
            dim=1))
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)
        x = self.upsample(x)  # size=(N, 256, 128, 128)

        layer0 = self.layer0_1x1(torch.cat(
            (layer0, message.expand(h // 2, w // 2, batch_size, self.message_size).permute(2, 3, 0, 1).contiguous()),
            dim=1))
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)
        x = self.upsample(x)

        x = torch.cat([x, x_original], dim=1) # size=(N, 256, 128, 128)
        x = self.conv_original_size2(x)
        out = self.conv_last(x)

        return out # size=(N, 3, 256, 256)


class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()
        # self.dropout = nn.Dropout(p=0.9)  # d

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        # out = self.dropout(out)
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out