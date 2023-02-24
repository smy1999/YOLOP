import math
import os
import sys

import torch
import torch.nn as nn

sys.path.append(os.getcwd())
from lib.utils import initialize_weights
from lib.models.common import Conv, SPP, BottleneckCSP, Focus, Concat
from torch.nn import Upsample
from lib.core.evaluate import SegmentationMetric

YOLOP = [
    [25],
    [-1, Focus, [3, 32, 3]],  # 0
    [-1, Conv, [32, 64, 3, 2]],  # 1
    [-1, BottleneckCSP, [64, 64, 1]],  # 2
    [-1, Conv, [64, 128, 3, 2]],  # 3
    [-1, BottleneckCSP, [128, 128, 3]],  # 4
    [-1, Conv, [128, 256, 3, 2]],  # 5
    [-1, BottleneckCSP, [256, 256, 3]],  # 6
    [-1, Conv, [256, 512, 3, 2]],  # 7
    [-1, SPP, [512, 512, [5, 9, 13]]],  # 8
    [-1, BottleneckCSP, [512, 512, 1, False]],  # 9
    [-1, Conv, [512, 256, 1, 1]],  # 10
    [-1, Upsample, [None, 2, 'nearest']],  # 11
    [[-1, 6], Concat, [1]],  # 12
    [-1, BottleneckCSP, [512, 256, 1, False]],  # 13
    [-1, Conv, [256, 128, 1, 1]],  # 14
    [-1, Upsample, [None, 2, 'nearest']],  # 15
    [[-1, 4], Concat, [1]],  # 16         #Encoder

    [16, Conv, [256, 128, 3, 1]],  # 34
    [-1, Upsample, [None, 2, 'nearest']],  # 35
    [-1, BottleneckCSP, [128, 64, 1, False]],  # 36
    [-1, Conv, [64, 32, 3, 1]],  # 37
    [-1, Upsample, [None, 2, 'nearest']],  # 38
    [-1, Conv, [32, 16, 3, 1]],  # 39
    [-1, BottleneckCSP, [16, 8, 1, False]],  # 40
    [-1, Upsample, [None, 2, 'nearest']],  # 41
    [-1, Conv, [8, 2, 3, 1]]  # 42 Lane line segmentation head
]


class MCnet(nn.Module):
    def __init__(self, block_cfg, **kwargs):
        super(MCnet, self).__init__()
        layers, save = [], []
        self.nc = 1
        self.detector_index = -1
        self.seg_out_idx = block_cfg[0][0:]

        # Build model
        for i, (from_, block, args) in enumerate(block_cfg[1:]):
            block = eval(block) if isinstance(block, str) else block  # eval strings
            block_ = block(*args)
            block_.index, block_.from_ = i, from_
            layers.append(block_)
            save.extend(x % i for x in ([from_] if isinstance(from_, int) else from_) if x != -1)  # append to savelist

        self.model, self.save = nn.Sequential(*layers), sorted(save)
        self.names = [str(i) for i in range(self.nc)]

        initialize_weights(self)

    def forward(self, x):
        cache = []
        out = []
        for i, block in enumerate(self.model):
            if block.from_ != -1:
                x = cache[block.from_] if isinstance(block.from_, int) else [x if j == -1 else cache[j] for j in
                                                                             block.from_]  # calculate concat detect
            x = block(x)
            if i in self.seg_out_idx:  # save driving area segment result
                m = nn.Sigmoid()
                out.append(m(x))
            cache.append(x if block.index in self.save else None)
        return out

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        # m = self.model[-1]  # Detect() module
        m = self.model[self.detector_index]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


def get_net(cfg, **kwargs):
    m_block_cfg = YOLOP
    model = MCnet(m_block_cfg, **kwargs)
    return model


if __name__ == "__main__":

    model = get_net(False)
    input_ = torch.randn((1, 3, 256, 256))
    gt_ = torch.rand((1, 2, 256, 256))
    metric = SegmentationMetric(2)
    model_out, SAD_out = model(input_)
    detects, dring_area_seg, lane_line_seg = model_out
    Da_fmap, LL_fmap = SAD_out
    for det in detects:
        print(det.shape)
    print(dring_area_seg.shape)
    print(lane_line_seg.shape)
