from torch import nn
import torch.nn.functional as F


class Rank(nn.Module):

    def __init__(self, conv_strides, pool_size, pool_strides):
        super(Rank, self).__init__()
        self.conv = nn.Conv2d(3, 96, kernel_size=8, stride=conv_strides)
        self.pool = nn.MaxPool2d(pool_size, stride=pool_strides)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size()[0], -1)
        return F.normalize(x, p=2, dim=1)
