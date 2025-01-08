import numpy as np
import torch
from torch import nn

__all__ = ['Loss_PhyHysLSTM']


class Loss_PhyHysLSTM(nn.Module):
    def __init__(self, Dif=None, hys=None):
        super(Loss_PhyHysLSTM, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.Dif = Dif
        self.hys = hys

    def forward(self, ag_all, pred_data_all, true_data, is_start=None):
        ag, ag_c = ag_all[0], ag_all[1]
        pred_data, pred_data_c = pred_data_all[0], pred_data_all[1]
        # data loss
        u_true, u_pred = true_data[:, :, 0 * 3:1 * 3], pred_data[:, :, 0 * 3:1 * 3]
        v_true, v_pred = true_data[:, :, 1 * 3:2 * 3], pred_data[:, :, 1 * 3:2 * 3]
        data_loss_u = self.mse_loss(u_true, u_pred)
        data_loss_v = self.mse_loss(v_true, v_pred)
        data_loss = (data_loss_u + data_loss_v) / 2
        # u, udot, v, vdot, rdot1, rdot2, g, ar
        u_c_pred = pred_data_c[:, :, 0 * 3:1 * 3]
        udot_c_pred = pred_data_c[:, :, 1 * 3:2 * 3]
        v_c_pred = pred_data_c[:, :, 2 * 3:3 * 3]
        vdot_c_pred = pred_data_c[:, :, 3 * 3:4 * 3]
        rdot1_c_pred = pred_data_c[:, :, 4 * 3:5 * 3]
        rdot2_c_pred = pred_data_c[:, :, 5 * 3:6 * 3]
        g_c_pred = pred_data_c[:, :, 6 * 3:7 * 3]
        ar2_c_pred, e1_c = self.hys.integrate(ag_c.repeat(1, 1, 3),
                                             u_c_pred, udot_c_pred, vdot_c_pred)
        PhyLoss_v = self.mse_loss(udot_c_pred, v_c_pred)
        PhyLoss_e1 = self.mse_loss(-ag_c - vdot_c_pred, g_c_pred)
        PhyLoss_e2 = self.mse_loss(e1_c, ar2_c_pred)
        PhyLoss_r = self.mse_loss(rdot1_c_pred, rdot2_c_pred)
        PhyLoss = (PhyLoss_v + 0.0002 * PhyLoss_e2 + 0.1*PhyLoss_e1 + PhyLoss_r) / 4
        loss = (PhyLoss + data_loss) / 2
        return loss

