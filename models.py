import torch
import torch.nn as nn
from torchvision import models

# class block(nn.Module):
#     def __init__(self, inplanes, planes, identity_downsample=None, stride=1):
#         super(block, self).__init__()
#         self.expansion = 4
#         self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, stride=1, padding=0)
#         self.bn1 = nn.BatchNorm1d(planes)
#         self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride, padding=1)
#         self.bn2 = nn.BatchNorm1d(planes)
#         self.conv3 = nn.Conv1d(planes, planes*self.expansion, kernel_size=1, stride=1, padding=0)
#         self.bn3 = nn.BatchNorm1d(planes*self.expansion)
#         self.relu = nn.ReLU()
#         self.identity_downsample = identity_downsample

#     def forward(self, x):
#         identity = x
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
#         x = self.conv3(x)
#         x = self.bn3(x)
#         if self.identity_downsample is not None:
#             identity = self.identity_downsample(identity)
#         x += identity
#         x = self.relu(x)
#         return x    
    
# class EcgResNet(nn.Module):
#     def __init__(self, block, layers, input_channels, inplanes=64, num_classes=9):
#         super(EcgResNet, self).__init__()
#         self.inplanes = inplanes
#         self.conv1 = nn.Conv1d(input_channels, self.inplanes, kernel_size=7, stride=2, padding=3)
#         self.bn1 = nn.BatchNorm1d(inplanes)
#         self.relu = nn.ReLU()
#         self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

#         # ResNet layers
#         self.layer1 = self._make_layer(block, layers[0], planes=64, stride=1)
#         self.layer2 = self._make_layer(block, layers[1], planes=128, stride=2)
#         self.layer3 = self._make_layer(block, layers[2], planes=256, stride=2)
#         self.layer4 = self._make_layer(block, layers[3], planes=512, stride=2)

#         self.avgpool = nn.AdaptiveAvgPool1d(1)
#         self.fc = nn.Linear(512*4, num_classes)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         x = self.avgpool(x)
#         x = x.reshape(x.shape[0], -1)
#         x = self.fc(x)

#         return x



#     def _make_layer(self, block, num_residual_blocks, planes, stride):
#         identity_downsample = None
#         layers = []

#         if stride != 1 or self.inplanes != planes * 4:
#             identity_downsample = nn.Sequential(nn.Conv1d(self.inplanes, planes*4, kernel_size=1, stride=stride),
#                                                 nn.BatchNorm1d(planes*4))
#         layers.append(block(self.inplanes, planes, identity_downsample, stride))
#         self.inplanes = planes * 4

#         for i in range(num_residual_blocks - 1):
#             layers.append(block(self.inplanes, planes))

#         return nn.Sequential(*layers)
    
# def ResNet50_1d(input_channels=1, num_classes = 8):
#     return EcgResNet(block, [3, 4, 6, 3], input_channels, num_classes)

def conv_block(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=17,
        stride=stride,
        padding=8,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv_subsampling(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv_block(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_block(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.dropout = nn.Dropout()
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(x))
        out = self.dropout(out)
        out = self.conv1(out)

        out = self.relu(self.bn2(out))
        out = self.dropout(out)
        out = self.conv2(out)

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        return out

class ResNet34_1d(nn.Module):
    def __init__(self, layers=(1, 5, 5, 5), num_classes=1000, replace_stride_with_dilation=None):
        super(ResNet34_1d, self).__init__()
        self.inplanes = 32
        self.dilation = 1

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]

        self.conv1 = conv_block(1, self.inplanes)
        self.bn1 = nn.BatchNorm1d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

        self._initialize_weights()

    def _make_layer(self, planes, blocks, stride=1, dilate=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                conv_subsampling(self.inplanes, planes * BasicBlock.expansion, stride),
                nn.BatchNorm1d(planes * BasicBlock.expansion),
            )

        layers = [BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def ResNet34_2d(num_classes=8):
    model = models.resnet34()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model