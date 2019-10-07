import sys

import torch
from torch.nn.modules.loss import _Loss


class TripletLoss(_Loss):

    def __init__(self, batch_size, device):
        super(TripletLoss, self).__init__()
        self.batch_size = batch_size
        self.device = device

    def forward(self, pred, target):
        # Clamp the input tensor between {epsilon, 1 - epsilon}
        pred = torch.clamp(pred, sys.float_info.epsilon, 1.0 - sys.float_info.epsilon)
        loss, g = torch.FloatTensor([0.0]).to(self.device), torch.FloatTensor([1.0]).to(self.device)

        for i in range(0, self.batch_size, 3):
            query_embedding = pred[i]
            positive_embedding = pred[i + 1]
            negative_embedding = pred[i + 2]

            # Calculate the hinge loss for the positive target
            dqp = torch.sqrt(torch.sum((query_embedding - positive_embedding) ** 2))
            # Calculate the hinge loss for the negative target.
            dqn = torch.sqrt(torch.sum((query_embedding - negative_embedding) ** 2))

            # Calculate the triplet loss, maximise positive, minimise negative.
            loss = loss + g + dqp - dqn

        return torch.max(
            torch.FloatTensor([0.0]).to(self.device),
            loss / torch.FloatTensor([self.batch_size / 3]).to(self.device)
        )
