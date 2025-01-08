# -*- coding: utf-8 -*-
import torch


class Hysteresis_Asym_MDOF_NewMark:
    def __init__(self, dt, device):
        self.device = device
        self.sn, self.pn = [(2, 5), (6, 7)], [(1, 1)]
        self.m = torch.as_tensor(1e3 * torch.tensor([270*3, 180, 90, 90, 90, 180, 90]),
                                 dtype=torch.float64, device=self.device)
        self.fy = torch.as_tensor(1e3 * torch.tensor([1225, 975, 490, 490, 490, 975, 490]),
                                  dtype=torch.float64, device=self.device)
        self.k0 = torch.as_tensor(1e6 * torch.tensor([245, 195, 98, 98, 98, 195, 98]),
                                  dtype=torch.float64, device=self.device)
        self.alpha = torch.as_tensor(torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
                                     dtype=torch.float64, device=self.device)
        self.dt = torch.as_tensor(dt, dtype=torch.float64, device=self.device)
        self.freedom = len(self.m)
        self.M = torch.diag(self.m)
        self.K0 = self.stiffnessSP(self.k0)
        # 预计算逆矩阵
        self.inv_M = torch.linalg.inv(self.M)
        eig_val, eig_vec = torch.linalg.eig(torch.inverse(self.M) @ self.K0)
        w_order = torch.argsort(torch.sqrt(torch.real(eig_val)))
        w = torch.sqrt(torch.real(eig_val[w_order]))
        A = 2 * 0.05 / (w[0] + w[1]) * torch.tensor([[w[0] * w[1]], [1]],
                                                    dtype=torch.float64,
                                                    device=device)
        self.C = A[0] * self.M + A[1] * self.K0


    def integrate(self, ag, prev_u, prev_v, prev_a):
        ag = ag.permute(0, 2, 1).repeat(1, 3, 1).to(self.device)
        n, num_steps = ag.size(0), ag.size(2)
        u = prev_u.permute(0, 2, 1).to(torch.float64).to(self.device)
        v = prev_v.permute(0, 2, 1).to(torch.float64).to(self.device)
        a = prev_a.permute(0, 2, 1).to(torch.float64).to(self.device)

        Rs = torch.zeros_like(ag, device=self.device, dtype=torch.float64)

        for i in range(1, num_steps):
            Rtrial, ktrial = self.state_determine_nmdof(Rs[:, :, i-1], u[:, :, i-1], u[:, :, i])
            Rs[:, :, i] = Rtrial
        ar = self.inv_M[2:5, 2:5].unsqueeze(0) @ Rs
        e1 = -ag - a - (self.C[3:4, 2:5] @ self.inv_M[2:5, 2:5]).unsqueeze(0)@v

        return (ar[:, 1:-1, :].permute(0, 2, 1).to(torch.float32),
                e1[:, 1:-1, :].permute(0, 2, 1).to(torch.float32))

    def state_determine_nmdof(self, last_rs, last_disp, disp):
        fy = self.fy[2:5]
        alpha = self.alpha[2:5]
        k0 = self.k0[2:5]
        force = torch.zeros_like(last_rs)
        last_shear = torch.cumsum(last_rs.flip(1), dim=1).flip(1)
        last_relative_disp = torch.cat((last_disp[:, 0:1], torch.diff(last_disp, dim=1)), dim=1)
        relative_disp = torch.cat((disp[:, 0:1], torch.diff(disp, dim=1)), dim=1)

        fy1_minusb = fy * (1 - alpha)
        ksh = alpha * k0
        d_disp = relative_disp - last_relative_disp

        c1 = ksh * relative_disp
        c2 = fy1_minusb * torch.ones_like(relative_disp)
        c3 = fy1_minusb * torch.ones_like(relative_disp)
        c = last_shear + k0.unsqueeze(0) * d_disp
        shear = torch.clamp(c1 - c2, min=torch.min(c1 + c3, c))
        condition = (torch.abs(shear - c) < 1e-3).double()
        stiffness = condition * k0 + (1 - condition) * ksh

        force[:, :-1] = shear[:, :-1] - shear[:, 1:]
        force[:, -1] = shear[:, -1]

        return force, stiffness

    def stiffnessSP(self, k):
        """
        Calculate the stiffness matrix for planar shear-type structures (series-parallel).

        Parameters:
        - k: Tensor of shear stiffness for each layer, can be 1D or 2D tensor.
        - sn: List of tuples (start, end) defining the start and end layers for series connections.
        - pn: List of tuples (start, end) defining the start and end layers for parallel connections.

        Returns:
        - K: Generated stiffness matrix, an n*n tensor or multi-dimensional tensor.
        """

        if len(k.shape) == 1:
            freedom = k.shape[0]
            K = torch.zeros((freedom, freedom), device=k.device, dtype=k.dtype)
        elif len(k.shape) == 2:
            freedom = k.shape[1]
            K = torch.zeros((len(k), freedom, freedom), device=k.device, dtype=k.dtype)
        else:
            raise ValueError("Unsupported shape of k")

        fsn = [s[0] for s in self.sn]
        lsn = [s[1] for s in self.sn]
        fpn = [p[0] for p in self.pn]
        lpn = [p[1] for p in self.pn]

        def fill_k_matrix(K, k, start, end):
            if len(k.shape) == 1:
                K[start, start] = k[start] + k[start + 1]
                K[start, start + 1] = -k[start + 1]
                for i in range(start + 1, end):
                    K[i, i] = k[i] + k[i + 1]
                    K[i, i - 1] = -k[i]
                    K[i, i + 1] = -k[i + 1]
                K[end, end] = k[end]
                K[end, end - 1] = -k[end]
            else:
                K[:, start, start] = k[:, start] + k[:, start + 1]
                K[:, start, start + 1] = -k[:, start + 1]
                for i in range(start + 1, end):
                    K[:, i, i] = k[:, i] + k[:, i + 1]
                    K[:, i, i - 1] = -k[:, i]
                    K[:, i, i + 1] = -k[:, i + 1]
                K[:, end, end] = k[:, end]
                K[:, end, end - 1] = -k[:, end]

        if fpn[0] == 1:
            fill_k_matrix(K, k, 0, lpn[0] - 1)
            for j in range(len(self.sn)):
                if len(k.shape) == 1:
                    K[lpn[0] - 1, lpn[0] - 1] += k[fsn[j] - 1]
                    K[lpn[0] - 1, fsn[j] - 1] = -k[fsn[j] - 1]
                    K[fsn[j] - 1, lpn[0] - 1] = -k[fsn[j] - 1]
                    fill_k_matrix(K, k, fsn[j] - 1, lsn[j] - 1)
                else:
                    K[:, lpn[0] - 1, lpn[0] - 1] += k[:, fsn[j] - 1]
                    K[:, lpn[0] - 1, fsn[j] - 1] = -k[:, fsn[j] - 1]
                    K[:, fsn[j] - 1, lpn[0] - 1] = -k[:, fsn[j] - 1]
                    fill_k_matrix(K, k, fsn[j] - 1, lsn[j] - 1)
        else:
            K[lpn[0] - 1, lpn[0] - 1] = k[lpn[0] - 1] if len(k.shape) == 1 else k[:, lpn[0] - 1]
            for j in range(len(self.sn)):
                fill_k_matrix(K, k, fsn[j] - 1, lsn[j] - 1)
                if fpn[0] > lsn[j]:
                    if len(k.shape) == 1:
                        K[lsn[j] - 1, lsn[j] - 1] += k[fpn[0] - 1]
                        K[lsn[j] - 1, fpn[0] - 1] = -k[fpn[0] - 1]
                        K[fpn[0] - 1, lsn[j] - 1] = -k[fpn[0] - 1]
                    else:
                        K[:, lsn[j] - 1, lsn[j] - 1] += k[:, fpn[0] - 1]
                        K[:, lsn[j] - 1, fpn[0] - 1] = -k[:, fpn[0] - 1]
                        K[:, fpn[0] - 1, lsn[j] - 1] = -k[:, fpn[0] - 1]
                else:
                    if len(k.shape) == 1:
                        K[lsn[j] - 1, lsn[j] - 1] = k[lsn[j] - 1]
                    else:
                        K[:, lsn[j] - 1, lsn[j] - 1] = k[:, lsn[j] - 1]

                fill_k_matrix(K, k, fpn[0] - 1, lpn[0] - 1)
                if fpn[0] > lsn[j]:
                    if len(k.shape) == 1:
                        K[fpn[0] - 1, fpn[0] - 1] += k[lsn[j] - 1]
                    else:
                        K[:, fpn[0] - 1, fpn[0] - 1] += k[:, lsn[j] - 1]
                else:
                    if len(k.shape) == 1:
                        K[lpn[0] - 1, lpn[0] - 1] += k[fsn[j] - 1]
                        K[lpn[0] - 1, fsn[j] - 1] = -k[fsn[j] - 1]
                        K[fsn[j] - 1, lpn[0] - 1] = -k[fsn[j] - 1]
                    else:
                        K[:, lpn[0] - 1, lpn[0] - 1] += k[:, fsn[j] - 1]
                        K[:, lpn[0] - 1, fsn[j] - 1] = -k[:, fsn[j] - 1]
                        K[:, fsn[j] - 1, lpn[0] - 1] = -k[:, fsn[j] - 1]

        return K