import torch
from torch import nn
from model.TripletConvNet import TripletConvNet


class DeepRank(nn.Module):

    def __init__(self):
        super(DeepRank, self).__init__()
        self.q_rank = TripletConvNet()
        self.p_rank = TripletConvNet()
        self.n_rank = TripletConvNet()

    def forward(self, x):
        query = self.q_rank(x[:, 0])
        positive = self.p_rank(x[:, 1])
        negative = self.n_rank(x[:, 2])
        return torch.stack([query, positive, negative])
