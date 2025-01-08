import numpy as np
import torch
from torch import nn

__all__ = ['Loss_PhyHysLSTM']


class HysU_RS:
    """
    用于输入位移响应，得到双线性本构单自由度体系系统的恢复力响应

    m = 1
    kesi = 0.05
    fy = 0.98
    alpha = 0.1
    dt = 0.05
    T = 1

    mat = sio.loadmat('data/response_blsdof.mat')
    print(mat.keys())
    ug = torch.tensor(mat['input'], dtype=torch.float32, device='cuda')
    u_prev = torch.tensor(mat['u_hys'], dtype=torch.float32, device='cuda')

    integrator = NewmarkIntegration(m, T, fy, alpha, dt, kesi)
    u, v, a, absa, Rs, k = integrator.integrate(ug, u_prev)

    plt.figure()
    plt.plot(np.array(u_prev.cpu())[0], np.array(Rs.cpu())[0])
    plt.show()
    """

    def __init__(self, m, T, fy, alpha, dt, kesi):
        self.m = m
        self.T = T
        self.fy = fy
        self.alpha = alpha
        self.dt = dt
        self.kesi = kesi

    def state_determine_nsdof(self, k1, last_r, last_u, u):
        fyl_minus_b = self.fy * (1 - self.alpha)
        fyl_minus_b = torch.full_like(k1, fyl_minus_b)  # 扩展张量的维度以匹配k1的维度
        k2 = self.alpha * k1
        d_u = u - last_u

        c1 = k2 * u
        c2 = 1.0 * fyl_minus_b
        c3 = 1.0 * fyl_minus_b
        c = last_r + k1 * d_u
        R = torch.max(c1 - c2, torch.min(c1 + c3, c))  # 使用torch.max和torch.min进行向量化操作

        return R

    def integrate(self, u_prev):
        u_prev = u_prev.squeeze()
        device = u_prev.device
        m = torch.full((u_prev.size(0),), self.m, device=device, dtype=torch.float64)
        k1 = (4 * torch.pi ** 2 * m / self.T ** 2).to(torch.float64)

        u = u_prev.to(torch.float64)
        Rs = torch.zeros_like(u_prev, device=device, dtype=torch.float64)

        for i in range(1, u_prev.size(1)):
            u0 = u[:, i - 1]
            R0 = Rs[:, i - 1]
            u1 = u[:, i]
            R1 = self.state_determine_nsdof(k1, R0, u0, u1)
            Rs[:, i] = R1
        ar = Rs / m[0]
        return ar.unsqueeze(-1).to(torch.float32)


class Loss_PhyHysLSTM(nn.Module):
    def __init__(self, Dif=None):
        super(Loss_PhyHysLSTM, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.Dif = Dif
        self.c = 0.628318530717959
        self.hys = HysU_RS(m=1, T=1, fy=0.98, alpha=0.1, dt=0.02, kesi=0.05)

        # 初始化权重为 None
        self.data_v_weight = None
        self.phy_e_weight = None
        self.phy_ar_weight = None
        self.phy_r_weight = None

    def forward(self, ag_all, pred_data_all, true_data):
        ag, ag_c = ag_all[0], ag_all[1]
        pred_data, pred_data_c = pred_data_all[0], pred_data_all[1]

        # 数据损失
        u_true, u_pred = true_data[:, :, 0:1], pred_data[:, :, 0:1]
        v_true, v_pred = true_data[:, :, 1:2], pred_data[:, :, 1:2]
        data_loss_u = self.mse_loss(u_true, u_pred)
        data_loss_v = self.mse_loss(v_true, v_pred)

        u_c_pred = pred_data_c[:, :, 0:1]
        udot_c_pred = pred_data_c[:, :, 1:2]
        v_c_pred = pred_data_c[:, :, 2:3]
        vdot_c_pred = pred_data_c[:, :, 3:4]
        ar_c_pred = pred_data_c[:, :, 4:5]
        ar2_c_pred = self.hys.integrate(u_c_pred)
        rdot1_c_pred = pred_data_c[:, :, 5:6]
        rdot2_c_pred = pred_data_c[:, :, 6:7]

        # 物理损失
        PhyLoss_v = self.mse_loss(udot_c_pred, v_c_pred)
        PhyLoss_ar = self.mse_loss(ar_c_pred, ar2_c_pred)
        PhyLoss_e = self.mse_loss(-ag_c - vdot_c_pred - self.c * v_c_pred, ar_c_pred)
        PhyLoss_r = self.mse_loss(rdot1_c_pred, rdot2_c_pred)

        # 第一次调用时，计算权重
        if self.data_v_weight is None:
            self.data_v_weight = (data_loss_u / (data_loss_v + 1e-8)).item()
            self.phy_e_weight = (PhyLoss_v / (PhyLoss_e + 1e-8)).item()
            self.phy_ar_weight = (PhyLoss_v / (PhyLoss_ar + 1e-8)).item()
            self.phy_r_weight = (PhyLoss_v / (PhyLoss_r + 1e-8)).item()

        # 使用计算出的权重进行缩放
        data_loss = (data_loss_u + self.data_v_weight * data_loss_v) / 2
        PhyLoss = (PhyLoss_v + self.phy_e_weight * PhyLoss_e +
                   self.phy_r_weight * PhyLoss_r + self.phy_ar_weight * PhyLoss_ar) / 4

        # 最终损失
        loss = (PhyLoss + data_loss) / 2
        return loss
