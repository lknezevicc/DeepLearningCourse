import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, in_channels, out_channels, stride=1, downsample=None):
    super(BasicBlock, self).__init__()
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                          stride=stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                          stride=1, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(out_channels)
    self.downsample = downsample
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    identity = x

    out = self.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))

    if self.downsample:
        identity = self.downsample(x)

    out += identity
    out = self.relu(out)

    return out


class CustomResNet(nn.Module):
  def __init__(self, block, layers, num_classes=10):
    super(CustomResNet, self).__init__()
    self.in_channels = 64

    self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)

    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
    self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(512 * block.expansion, num_classes)

  def _make_layer(self, block, out_channels, blocks, stride=1):
    downsample = None
    if stride != 1 or self.in_channels != out_channels * block.expansion:
      downsample = nn.Sequential(
          nn.Conv2d(self.in_channels, out_channels * block.expansion, 
                  kernel_size=1, stride=stride, bias=False),
          nn.BatchNorm2d(out_channels * block.expansion),
      )

    layers = [block(self.in_channels, out_channels, stride, downsample)]
    self.in_channels = out_channels * block.expansion
    for _ in range(1, blocks):
      layers.append(block(self.in_channels, out_channels))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.relu(self.bn1(self.conv1(x)))

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.fc(x)

    return x


def resnet(num_classes=10):
  return CustomResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)