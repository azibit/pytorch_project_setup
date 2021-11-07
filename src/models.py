"""
ResNet in PyTorch

BasicBlock and Bottleneck module is from the original ResNet paper:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv: 1512.03385

PreActBlock and PreActBottleneck module is from the later paper:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027

"""

# PyTorch is an open source machine learning library used for developing and training neural network based deep learning models.
import torch

# Contains the functions with which to build Neural Networks
import torch.nn as nn

# torch.nn is stateful. torch.nn.Functional is stateless.
import torch.nn.functional as F

from nn import BatchNorm2d, Sequential

# A Variable wraps a Tensor
from torch.autograd import Variable

def conv3x3(in_planes, out_planes, stride = 1):
    return nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = stride, padding = 1, bias = False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride = 1):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = BatchNorm2d(planes)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes)

        self.shortcut = Sequential()

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                Conv2d(in_planes, self.expansion * planes, kernel_size = 1, stride = stride, bias = False),
                BatchNorm2d(self.expansion * planes)
            )

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out =
