import torch as t
from torch import nn


class Conv3X1X1(nn.Module):

    def __init__(self, in_channels, out_channels, stride, padding, is_relu):
        super(Conv3X1X1, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 1, 1), stride=stride, padding=padding, bias=False),
            nn.BatchNorm3d(num_features=out_channels)
        )
        if is_relu:
            self.block.add_module("relu", nn.ReLU())

    def forward(self, x):
        return self.block(x)


class Conv1X1X1(nn.Module):

    def __init__(self, in_channels, out_channels, stride, padding, is_relu):
        super(Conv1X1X1, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1, 1), stride=stride, padding=padding, bias=False),
            nn.BatchNorm3d(num_features=out_channels)
        )
        if is_relu:
            self.block.add_module("relu", nn.ReLU())

    def forward(self, x):
        return self.block(x)


class Conv1X3X3(nn.Module):

    def __init__(self, in_channels, out_channels, stride, padding, is_relu):
        super(Conv1X3X3, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 3, 3), stride=stride, padding=padding, bias=False),
            nn.BatchNorm3d(num_features=out_channels)
        )
        if is_relu:
            self.block.add_module("relu", nn.ReLU())

    def forward(self, x):
        return self.block(x)


class BottleNeck(nn.Module):
    """
    3x3
    1x1
    3x3
    """

    def __init__(self, in_channels, out_channels, is_half, is_fastpath):
        super(BottleNeck, self).__init__()
        if is_half:
            stride = [[1, 1, 1], [1, 2, 2], [1, 1, 1]]
        else:
            stride = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        middle_channels = out_channels // 4
        if not is_fastpath:
            self.block = nn.Sequential(
                Conv1X1X1(in_channels=in_channels, out_channels=middle_channels, stride=stride[0], padding=(0, 0, 0), is_relu=False),
                Conv1X3X3(in_channels=middle_channels, out_channels=middle_channels, stride=stride[1], padding=(0, 1, 1), is_relu=False),
                Conv1X1X1(in_channels=middle_channels, out_channels=out_channels, stride=stride[2], padding=(0, 0, 0), is_relu=True)
            )
        else:
            self.block = nn.Sequential(
                Conv3X1X1(in_channels=in_channels, out_channels=middle_channels, stride=stride[0], padding=(1, 0, 0), is_relu=False),
                Conv1X3X3(in_channels=middle_channels, out_channels=middle_channels, stride=stride[1], padding=(0, 1, 1), is_relu=False),
                Conv1X1X1(in_channels=middle_channels, out_channels=out_channels, stride=stride[2], padding=(0, 0, 0), is_relu=True)
            )
        if in_channels != out_channels or is_half:
            self.downsample = Conv1X1X1(in_channels=in_channels, out_channels=out_channels, stride=stride[1], padding=(0, 0, 0), is_relu=True)
        self.is_downsample = in_channels != out_channels or is_half

    def forward(self, x):
        if self.is_downsample:
            result = self.downsample(x) + self.block(x)
            return result
        result = x + self.block(x)
        return result


def make_layer(bottle_neck_count, is_half, in_channels, out_channels, is_fastpath):
    """
    多个bottle_neck组合模块
    :param bottle_neck_count: bottle_neck的数目
    :param is_half: 模块输出空间尺寸是否缩小一半
    :param in_channels: 第一个bottle_neck的输入channels
    :param out_channels: 第一个bottle_neck的输出channels
    :param is_fastpath: 是否是FastPath
    :return:
    """
    if is_half:
        is_halfs = [True] + [False] * (bottle_neck_count - 1)
    else:
        is_halfs = [False] * bottle_neck_count
    in_channelses = [in_channels] + [out_channels] * (bottle_neck_count - 1)
    out_channelses = [out_channels] * bottle_neck_count
    block = nn.Sequential()
    for i in range(bottle_neck_count):
        in_channels = in_channelses[i]
        out_channels = out_channelses[i]
        is_half = is_halfs[i]
        block.add_module("%d" % (i,), BottleNeck(in_channels, out_channels, is_half, is_fastpath))
    return block


class SlowPath(nn.Module):

    def __init__(self, alpha):
        super(SlowPath, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False),
            nn.BatchNorm3d(num_features=64),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.layer1 = make_layer(3, False, in_channels=64, out_channels=256, is_fastpath=False)
        self.layer2 = make_layer(4, True, in_channels=256, out_channels=512, is_fastpath=False)
        self.layer3 = make_layer(6, True, in_channels=512, out_channels=1024, is_fastpath=False)
        self.layer4 = make_layer(3, True, in_channels=1024, out_channels=2048, is_fastpath=False)
        self.fusion_pool = nn.Conv3d(in_channels=8, out_channels=64, kernel_size=(5, 1, 1), stride=(alpha, 1, 1), padding=(2, 0, 0))
        self.fusion_layer1 = nn.Conv3d(in_channels=32, out_channels=256, kernel_size=(5, 1, 1), stride=(alpha, 1, 1), padding=(2, 0, 0))
        self.fusion_layer2 = nn.Conv3d(in_channels=64, out_channels=512, kernel_size=(5, 1, 1), stride=(alpha, 1, 1), padding=(2, 0, 0))
        self.fusion_layer3 = nn.Conv3d(in_channels=128, out_channels=1024, kernel_size=(5, 1, 1), stride=(alpha, 1, 1), padding=(2, 0, 0))

    def forward(self, x, fast_features):
        conv1_result = self.conv1(x)
        pool_result = self.pool1(conv1_result)
        layer1_result = self.layer1(pool_result + self.fusion_pool(fast_features[0]))
        layer2_result = self.layer2(layer1_result + self.fusion_layer1(fast_features[1]))
        layer3_result = self.layer3(layer2_result + self.fusion_layer2(fast_features[2]))
        layer4_result = self.layer4(layer3_result + self.fusion_layer3(fast_features[3]))
        return layer4_result


class FastPath(nn.Module):

    def __init__(self):
        super(FastPath, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=8, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(num_features=8),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.layer1 = make_layer(3, False, in_channels=8, out_channels=32, is_fastpath=True)
        self.layer2 = make_layer(4, True, in_channels=32, out_channels=64, is_fastpath=True)
        self.layer3 = make_layer(6, True, in_channels=64, out_channels=128, is_fastpath=True)
        self.layer4 = make_layer(3, True, in_channels=128, out_channels=256, is_fastpath=True)

    def forward(self, x):
        conv1_result = self.conv1(x)
        pool_result = self.pool1(conv1_result)
        layer1_result = self.layer1(pool_result)
        layer2_result = self.layer2(layer1_result)
        layer3_result = self.layer3(layer2_result)
        layer4_result = self.layer4(layer3_result)
        return layer4_result, [pool_result, layer1_result, layer2_result, layer3_result]


class SlowFastNet(nn.Module):

    def __init__(self, num_classes, slow_tao, alpha):
        """

        :param num_classes: 类别数目
        :param slow_tao: slowpath的帧采样步长
        :param alpha: slowpath的帧采样步长与fastpath帧采样步长的比值
        """
        super(SlowFastNet, self).__init__()
        self.slow_tao = slow_tao
        self.fast_tao = slow_tao // alpha
        self.slow_path = SlowPath(alpha)
        self.fast_path = FastPath()
        self.avg_pool = nn.AdaptiveAvgPool3d(output_size=1)
        self.clsf = nn.Linear(in_features=256 + 2048, out_features=num_classes)

    def forward(self, x):
        slow_input = x[:, :, ::self.slow_tao, :, :]
        fast_input = x[:, :, ::self.fast_tao, :, :]
        print(slow_input.size())
        print(fast_input.size())
        fast_result, fast_features = self.fast_path(fast_input)
        slow_result = self.slow_path(slow_input, fast_features)
        slow_global_pool = self.avg_pool(slow_result).view((x.size()[0], -1))
        fast_global_pool = self.avg_pool(fast_result).view((x.size()[0], -1))
        cat_result = t.cat([slow_global_pool, fast_global_pool], dim=1)
        result = self.clsf(cat_result)
        return result


if __name__ == "__main__":
    d = t.randn(2, 3, 64, 224, 224)
    model = SlowFastNet(num_classes=10, slow_tao=16, alpha=8)
    out = model(d)
    print(out.size())


