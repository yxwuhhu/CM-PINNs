import numpy as np
import torch
from torch import nn

__all__ = ['Loss_PhyHysLSTM']


class Loss_PhyHysLSTM(nn.Module):  # 注意继承 nn.Module
    def __init__(self, is_run1=False, Dif=None):
        super(Loss_PhyHysLSTM, self).__init__()
        self.is_run1 = is_run1
        self.mse_loss = nn.MSELoss()
        self.Dif = Dif

    def forward(self, ag_all, pred_data_all, true_data):
        ag, ag_c = ag_all[0], ag_all[1]
        pred_data, pred_data_c = pred_data_all[0], pred_data_all[1]
        if self.is_run1:
            # data loss
            u_true, u_pred = true_data[:, :, 0:1], pred_data[:, :, 0:1]
            v_true, v_pred = true_data[:, :, 1:2], pred_data[:, :, 1:2]
            data_loss_u = self.mse_loss(u_true, u_pred)
            data_loss_v = self.mse_loss(v_true, v_pred)
            data_loss = data_loss_u + data_loss_v

            udot_c_pred = pred_data_c[:, :, 0:1]
            v_c_pred = pred_data_c[:, :, 1:2]
            vdot_c_pred = pred_data_c[:, :, 2:3]
            g_c_pred = pred_data_c[:, :, 3:4]
            rdot1_c_pred = pred_data_c[:, :, 4:5]
            rdot2_c_pred = pred_data_c[:, :, 5:6]

            PhyLoss_v = self.mse_loss(udot_c_pred, v_c_pred)
            PhyLoss_e = self.mse_loss(-ag_c - vdot_c_pred, g_c_pred)
            PhyLoss_r = self.mse_loss(rdot1_c_pred, rdot2_c_pred)
            PhyLoss = PhyLoss_v + PhyLoss_e + PhyLoss_r
            loss = PhyLoss + data_loss

        else:
            pred_data = pred_data_all
            data_loss_u = self.mse_loss(pred_data, true_data)
            loss = data_loss_u
        return loss
