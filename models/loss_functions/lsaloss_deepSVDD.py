import torch
import torch.nn as nn

from models.loss_functions.autoregression_loss import AutoregressionLoss
from models.loss_functions.reconstruction_loss import ReconstructionLoss
from models.loss_functions.deepSVDD_loss import DeepSVDDLoss

class LSALoss_deepSVDD(nn.Module):
    """
    Implements the loss of a LSA model.
    It is a sum of the reconstruction loss and the autoregression loss.
    """
    def __init__(self, lam_rec=1, lam_svdd=0,  c=0, R=0, nu=0, objective='soft-boundary'):
        # type: (int, float) -> None
        """
        Class constructor.

        :param cpd_channels: number of bins in which the multinomial works.
        :param lam: weight of the autoregression loss.
        """
        super(LSALoss_deepSVDD, self).__init__()

        self.lam_rec = lam_rec
        self.lam_svdd = lam_svdd
        # 上面代码暂时留着不动，下面才是我的论文的核心
        self.c = c
        self.R = R
        self.nu = nu
        self.objective = objective


        # Set up loss modules
        self.reconstruction_loss_fn = ReconstructionLoss()
        self.deepSVDD_loss_fn = DeepSVDDLoss(c, R, nu, objective)

        # Numerical variables
        self.reconstruction_loss = None
        self.deepSVDD_loss = None
        self.total_loss = None

    def forward(self, x, x_r, z):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        """
        Forward propagation.

        :param x: the batch of input samples.
        :param x_r: the batch of reconstructions.
        :param z: the batch of latent representations.
        :param z_dist: the batch of estimated cpds.
        :return: the loss of the model (averaged along the batch axis).
        """
        # Compute pytorch loss
        rec_loss = self.reconstruction_loss_fn(x, x_r)
        dsvdd_loss = self.deepSVDD_loss_fn(z)
        # tot_loss = rec_loss + self.lam * dsvdd_loss
        tot_loss = self.lam_rec * rec_loss +  self.lam_svdd * dsvdd_loss

        # Store numerical
        self.reconstruction_loss = rec_loss.item()
        self.deepSVDD_loss = dsvdd_loss.item()
        self.total_loss = tot_loss.item()

        # 计算一个 losses_4_test, TODO

        return tot_loss