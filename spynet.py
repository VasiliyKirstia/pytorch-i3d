#!/usr/bin/env python

import torch
import torch.utils.serialization

import getopt
import math
from torch import nn
import numpy
import os
import PIL
import PIL.Image
import sys

arguments_strModel = 'sintel-final'
arguments_strFirst = './images/first.png'
arguments_strSecond = './images/second.png'
arguments_strOut = './out.flo'

for strOption, strArgument in \
getopt.getopt(sys.argv[1:], '', [strParameter[2:] + '=' for strParameter in sys.argv[1::2]])[0]:
    if strOption == '--model' and strArgument != '': arguments_strModel = strArgument  # which model to use, see below
    if strOption == '--first' and strArgument != '': arguments_strFirst = strArgument  # path to the first frame
    if strOption == '--second' and strArgument != '': arguments_strSecond = strArgument  # path to the second frame
    if strOption == '--out' and strArgument != '': arguments_strOut = strArgument  # path to where the output should be stored


# end

##########################################################

class Preprocess(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        r = (x[:, 0:1, :, :] - 0.406) / 0.225
        g = (x[:, 1:2, :, :] - 0.456) / 0.224
        b = (x[:, 2:3, :, :] - 0.485) / 0.229

        return torch.cat([r, g, b], 1)


class Basic(nn.Module):
    def __init__(self, level):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)
        )

    def forward(self, x):
        return self.model(x)


class Backward(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, flow):
        if (
            hasattr(self, 'tensorGrid') == False
            or self.tensor_grid.size(0) != flow.size(0)
            or self.tensor_grid.size(2) != flow.size(2)
            or self.tensor_grid.size(3) != flow.size(3)
        ):
            horizontal = (torch
                .linspace(-1.0, 1.0, flow.size(3))
                .view(1, 1, 1, flow.size(3))
                .expand(flow.size(0), -1, flow.size(2), -1))

            vertical = (torch
                .linspace(-1.0, 1.0, flow.size(2))
                .view(1, 1, flow.size(2), 1)
                .expand(flow.size(0), -1, -1, flow.size(3)))

            self.tensor_grid = torch.cat([horizontal, vertical], 1).cuda()  # fixme: find a way to dynamically detect model device

        flow = torch.cat([
            flow[:, 0:1, :, :] / ((x.size(3) - 1.0) / 2.0),
            flow[:, 1:2, :, :] / ((x.size(2) - 1.0) / 2.0),
        ], 1)

        return nn.functional.grid_sample(
            input=x,
            grid=(self.tensor_grid + flow).permute(0, 2, 3, 1),
            mode='bilinear',
            padding_mode='border'
        )


class SPyNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.module_preprocess = Preprocess()

        self.module_basic = nn.ModuleList([Basic(level) for level in range(6)])

        self.module_backward = Backward()

        self.load_state_dict(torch.load('./network-' + arguments_strModel + '.pytorch'))

    def forward(self, x1, x2):
        flow = []

        x1 = [self.module_preprocess(x1)]
        x2 = [self.module_preprocess(x2)]

        for level in range(5):
            if x1[0].size(2) > 32 or x1[0].size(3) > 32:
                x1.insert(0, nn.functional.avg_pool2d(input=x1[0], kernel_size=2, stride=2))
                x2.insert(0, nn.functional.avg_pool2d(input=x2[0], kernel_size=2, stride=2))

        flow = x1[0].new_zeros(
            x1[0].size(0), 2, int(math.floor(x1[0].size(2) / 2.0)), int(math.floor(x1[0].size(3) / 2.0)))

        for level in range(len(x1)):
            upsampled = nn.functional.interpolate(input=flow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0

            if upsampled.size(2) != x1[level].size(2):
                upsampled = nn.functional.pad(input=upsampled, pad=[0, 0, 0, 1], mode='replicate')
            if upsampled.size(3) != x1[level].size(3):
                upsampled = nn.functional.pad(input=upsampled, pad=[0, 1, 0, 0], mode='replicate')

            flow = self.module_basic[level](torch.cat([
                x1[level], self.module_backward(x2[level], upsampled), upsampled
            ], 1)) + upsampled

        return flow


moduleNetwork = SPyNet().cuda().eval()


##########################################################

def estimate(tensorFirst, tensorSecond):
    tensorOutput = torch.FloatTensor()

    assert (tensorFirst.size(1) == tensorSecond.size(1))
    assert (tensorFirst.size(2) == tensorSecond.size(2))

    intWidth = tensorFirst.size(2)
    intHeight = tensorFirst.size(1)

    assert (
                intWidth == 1024)  # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
    assert (
                intHeight == 416)  # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

    if True:
        tensorFirst = tensorFirst.cuda()
        tensorSecond = tensorSecond.cuda()
        tensorOutput = tensorOutput.cuda()
    # end

    if True:
        tensorPreprocessedFirst = tensorFirst.view(1, 3, intHeight, intWidth)
        tensorPreprocessedSecond = tensorSecond.view(1, 3, intHeight, intWidth)

        intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 32.0) * 32.0))
        intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 32.0) * 32.0))

        tensorPreprocessedFirst = torch.nn.functional.interpolate(input=tensorPreprocessedFirst,
                                                                  size=(intPreprocessedHeight, intPreprocessedWidth),
                                                                  mode='bilinear', align_corners=False)
        tensorPreprocessedSecond = torch.nn.functional.interpolate(input=tensorPreprocessedSecond,
                                                                   size=(intPreprocessedHeight, intPreprocessedWidth),
                                                                   mode='bilinear', align_corners=False)

        tensorFlow = torch.nn.functional.interpolate(
            input=moduleNetwork(tensorPreprocessedFirst, tensorPreprocessedSecond), size=(intHeight, intWidth),
            mode='bilinear', align_corners=False)

        tensorFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
        tensorFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

        tensorOutput.resize_(2, intHeight, intWidth).copy_(tensorFlow[0, :, :, :])
    # end

    if True:
        tensorFirst = tensorFirst.cpu()
        tensorSecond = tensorSecond.cpu()
        tensorOutput = tensorOutput.cpu()
    # end

    return tensorOutput


# end

##########################################################

if __name__ == '__main__':
    tensorFirst = torch.FloatTensor(
        numpy.array(PIL.Image.open(arguments_strFirst))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (
                    1.0 / 255.0))
    tensorSecond = torch.FloatTensor(
        numpy.array(PIL.Image.open(arguments_strSecond))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (
                    1.0 / 255.0))

    tensorOutput = estimate(tensorFirst, tensorSecond)

    objectOutput = open(arguments_strOut, 'wb')

    numpy.array([80, 73, 69, 72], numpy.uint8).tofile(objectOutput)
    numpy.array([tensorOutput.size(2), tensorOutput.size(1)], numpy.int32).tofile(objectOutput)
    numpy.array(tensorOutput.numpy().transpose(1, 2, 0), numpy.float32).tofile(objectOutput)

    objectOutput.close()
# end
