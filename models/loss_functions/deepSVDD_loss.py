import os,time
# Update network parameters via backpropagation: forward + backward + optimize
# loss.backward()
# optimizer.step()
#
# # Update hypersphere radius R on mini-batch distances
# if (self.objective == 'soft-boundary') and (epoch >= self.warm_up_n_epochs):
#     self.R.data = torch.tensor(get_radius(dist, self.nu), device=self.device)


import torch

from models.base import BaseModule


class DeepSVDDLoss(BaseModule):
    """
    Implements the reconstruction loss.
    """
    def __init__(self, c, R, nu, objective):
        # type: () -> None
        """
        Class constructor.
        """
        super(DeepSVDDLoss, self).__init__()
        #
        self.c = c
        self.R = R
        self.nu = nu
        self.objective = objective

    def forward(self, x):
        # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
        """
        Forward propagation.

        :param x: the batch of input samples.
        :param x_r: the batch of reconstructions.
        :return: the mean reconstruction loss (averaged along the batch axis).
        """

        dist = torch.sum((x - self.c) ** 2, dim=1)
        if self.objective == 'soft-boundary':
            scores = dist - self.R ** 2
            loss = self.R ** 2 + (1 / self.nu) * torch.mean(
                torch.max(torch.zeros_like(scores), scores))
        else:
            loss = torch.mean(dist)
        #
        # 计算一个 losses_4_test, TODO

        return loss
