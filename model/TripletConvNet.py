import torch
from torch import nn
from model.ConvNet import ConvNet
from model.Rank import Rank
import torch.nn.functional as F


class TripletConvNet(nn.Module):

    def __init__(self):
        super(TripletConvNet, self).__init__()
        self.conv_net = ConvNet()
        self.rank1 = Rank(conv_strides=(16, 16), pool_size=(3, 3), pool_strides=(4, 4))
        self.rank2 = Rank(conv_strides=(32, 32), pool_size=(7, 7), pool_strides=(2, 2))
        self.embedding = nn.Linear(in_features=5056, out_features=4096)

    def forward(self, x):
        rank_cat = torch.cat([self.rank1(x), self.rank2(x)], dim=1)
        x = torch.cat([rank_cat, self.conv_net(x)], dim=1)
        x = self.embedding(x)
        return F.normalize(x, dim=0)
