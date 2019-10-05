import sys

import torch
from torch.nn.modules.loss import _Loss


class TripletLoss(_Loss):

    def __init__(self, batch_size):
        super(TripletLoss, self).__init__()
        self.batch_size = batch_size

    def forward(self, pred, target):
        # Clamp the input tensor between {epsilon, 1 - epsilon}
        pred = torch.clamp(pred, sys.float_info.epsilon, 1.0 - sys.float_info.epsilon)
        loss, g = torch.FloatTensor([0.0]), torch.FloatTensor([1.0])

        for i in range(0, self.batch_size, 3):
            query_embedding = pred[i + 0]
            positive_embedding = pred[i + 1]
            negative_embedding = pred[i + 2]

            # Calculate the hinge loss for the positive target
            dqp = torch.sqrt(torch.sum((query_embedding - positive_embedding) ** 2))
            # Calculate the hinge loss for the negative target.
            dqn = torch.sqrt(torch.sum((query_embedding - negative_embedding) ** 2))

            # Calculate the triplet loss, maximise positive, minimise negative.
            loss = loss + g + dqp - dqn

        # When the batch size < 3 causes a divide-by-zero error
        norm = torch.FloatTensor([1.0]) if self.batch_size < 3 else torch.FloatTensor([self.batch_size / 3])
        return torch.max(
            torch.FloatTensor([0.0]),
            loss / norm
        )
