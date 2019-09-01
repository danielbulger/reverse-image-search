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
        loss, g = torch.tensor([0]), torch.tensor([1])

        for i in range(0, self.batch_size, 3):
            q_embedding = pred[i + 0]
            p_embedding = pred[i + 1]
            n_embedding = pred[i + 2]

            # Calculate the hinge loss for the positive target
            dqp = torch.sqrt(torch.sum((q_embedding - p_embedding) ** 2))

            # Calculate the hinge loss for the negative target.
            dqn = torch.sqrt(torch.sum((q_embedding - n_embedding) ** 2))

            loss = loss + g + dqp - dqn

        return torch.max(
            torch.tensor([0]),
            loss / (self.batch_size / 3)
        )
