import torch.nn as nn
from torch.nn import init
from seldnet_model import CRNN


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class ChannelAttention(nn.Module):
    def __init__(self, channel=10, expansion=2, num_layer=2):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.mlp = nn.Sequential()
        layers = [channel]
        layers += [channel * expansion] * num_layer
        layers += [channel]
        self.mlp.add_module(name='flatten', module=Flatten())
        for i in range(len(layers) - 2):
            self.mlp.add_module(name=f'linear_{i}', module=nn.Linear(in_features=layers[i], out_features=layers[i + 1]))
            self.mlp.add_module(name=f'bn_{i}', module=nn.BatchNorm1d(num_features=layers[i + 1]))
            self.mlp.add_module(name=f'relu_{i}', module=nn.ReLU())
        self.mlp.add_module(name='last_fc', module=nn.Linear(in_features=layers[-2], out_features=layers[-1]))

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.mlp(out)
        return out.unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    def __init__(self, channel=10, channel_reduction=2, num_layers=3, dia_val=2):
        super(SpatialAttention, self).__init__()
        self.mlp = nn.Sequential()
        self.mlp.add_module('conv_reduce_1',
                            nn.Conv2d(in_channels=channel, out_channels=channel // channel_reduction, kernel_size=1)
                            )
        self.mlp.add_module('bn_reduce_1', nn.BatchNorm2d(num_features=channel // channel_reduction))
        self.mlp.add_module('relu_reduce_1', nn.ReLU())
        for i in range(num_layers):
            self.mlp.add_module(f'conv_{i}',
                                nn.Conv2d(
                                    in_channels=channel // channel_reduction,
                                    out_channels=channel // channel_reduction,
                                    kernel_size=3,
                                    padding=2,
                                    dilation=dia_val
                                )
                                )
            self.mlp.add_module(f'bn_{i}', nn.BatchNorm2d(num_features=channel // channel_reduction))
            self.mlp.add_module(f'relu_{i}', nn.ReLU())
        self.mlp.add_module('last_conv_reduce',
                            nn.Conv2d(in_channels=channel // channel_reduction, out_channels=1, kernel_size=1)
                            )

    def forward(self, x):
        res = self.mlp(x)
        res = res.expand_as(x)
        return res


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.channel_attention = ChannelAttention()
        self.spatial_attention = SpatialAttention()
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        channel_map = self.channel_attention(x)
        spatial_map = self.spatial_attention(x)
        attention = self.sigmoid(channel_map.expand_as(x) + spatial_map)
        res = (1 + attention) * x
        return res, attention, channel_map, spatial_map


class EarlyAttention(nn.Module):
    def __init__(self, data_in, data_out, params):
        super(EarlyAttention, self).__init__()
        self.early_attention = Attention()
        self.downstream_task = CRNN(data_in, data_out, params)

    def forward(self, x):
        attention, channel_map, spatial_map = None, None, None
        x, attention, channel_map, spatial_map = self.early_attention(x)
        doa = self.downstream_task(x)
        return doa, attention, channel_map, spatial_map

