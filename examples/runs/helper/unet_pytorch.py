import torch
import torch.nn as nn
import torch.nn.functional as F


MODEL_UNET_2D = 'unet2d'


class ConvDONormReLu2D(nn.Sequential):

    def __init__(self, in_ch, out_ch, dropout_p: float = 0.0, norm: str = 'bn'):
        super().__init__()

        self.add_module('conv', nn.Conv2d(in_ch, out_ch, 3, padding=1))
        if dropout_p > 0:
            self.add_module('dropout', nn.Dropout2d(p=dropout_p, inplace=True))
        if norm == 'bn':
            self.add_module(norm, nn.BatchNorm2d(out_ch))
        # elif norm == 'ln':
        #     self.modules.add_module(norm, nn.LayerNorm(input.size()[1:]))

        self.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, x):
        return super().forward(x)


class DownConv2D(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_p: float = 0.0, norm: str = 'bn'):
        super().__init__()

        self.double_conv = nn.Sequential(ConvDONormReLu2D(in_ch, out_ch, dropout_p, norm),
                                         ConvDONormReLu2D(out_ch, out_ch, dropout_p, norm))
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip_x = self.double_conv(x)
        x = self.pool(skip_x)
        return x, skip_x


class UpConv2D(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_p: float = 0.0, norm: str = 'bn', transpose: bool = False):
        super().__init__()
        self.transpose = transpose

        if self.transpose:
            self.upconv = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        else:
            self.upconv = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.double_conv = nn.Sequential(ConvDONormReLu2D(2 * out_ch, out_ch, dropout_p, norm),
                                         ConvDONormReLu2D(out_ch, out_ch, dropout_p, norm))

    def forward(self, x, skip_x):
        if not self.transpose:
            x = torch.nn.functional.interpolate(x, mode='nearest', scale_factor=2)
        up = self.upconv(x)

        up_shape, skip_shape = up.size()[-2:], skip_x.size()[-2:]
        if up_shape < skip_shape:
            x_diff = skip_shape[-1] - up_shape[-1]
            y_diff = skip_shape[-2] - up_shape[-2]
            x_pad = (x_diff // 2, x_diff // 2 + (x_diff % 2))
            y_pad = (y_diff // 2, y_diff // 2 + (y_diff % 2))
            up = F.pad(up, x_pad + y_pad)

        x = torch.cat((up, skip_x), 1)
        x = self.double_conv(x)
        return x


class UNetModel(nn.Module):

    def __init__(self, ch_in: int, ch_out: int, n_channels: int = 32, n_pooling: int = 3,
                 dropout_p: float = 0.2, norm: str = 'bn', **kwargs):
        super().__init__()

        n_classes = ch_out
        ch_out = n_channels

        self.down_convs = nn.ModuleList()
        for i in range(n_pooling):
            down_conv = DownConv2D(ch_in, ch_out, dropout_p, norm)
            self.down_convs.append(down_conv)
            ch_in = ch_out
            ch_out *= 2

        self.bottleneck = nn.Sequential(ConvDONormReLu2D(ch_in, ch_out, dropout_p, norm),
                                        ConvDONormReLu2D(ch_out, ch_out, dropout_p, norm))

        self.up_convs = nn.ModuleList()
        for i in range(n_pooling, 0, -1):
            ch_in = ch_out
            ch_out = ch_in // 2
            up_conv = UpConv2D(ch_in, ch_out, dropout_p, norm)
            self.up_convs.append(up_conv)

        ch_in = ch_out
        self.conv_cls = nn.Conv2d(ch_in, n_classes, 1)

    def forward(self, x):
        skip_connections = []
        for down_conv in self.down_convs:
            x, skip_x = down_conv(x)
            skip_connections.append(skip_x)

        x = self.bottleneck(x)

        for inv_depth, up_conv in enumerate(self.up_convs, 1):
            skip_x = skip_connections[-inv_depth]
            x = up_conv(x, skip_x)

        logits = self.conv_cls(x)
        return logits