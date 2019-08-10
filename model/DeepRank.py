import torch
from torch import nn
from model.ConvNet import ConvNet
from model.Rank import Rank


class DeepRank(nn.Module):

    def __init__(self):
        super(DeepRank, self).__init__()
        self.convnet = ConvNet()
        self.rank1 = Rank(conv_strides=16, pool_size=3, pool_strides=4)
        self.rank2 = Rank(conv_strides=32, pool_size=7, pool_strides=2)

    def forward(self, x):
        conv_output = self.convnet(x)
        rank1_output = self.rank1(x)
        rank2_output = self.rank2(x)

        rank_cat = torch.cat(rank1_output, rank2_output)
        combine = torch.cat(rank_cat, conv_output)

        print(combine.size())

        return combine
