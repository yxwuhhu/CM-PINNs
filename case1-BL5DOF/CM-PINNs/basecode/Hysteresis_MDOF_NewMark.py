# -*- coding: utf-8 -*-
import numpy as np
import scipy.io as sio
import torch
import matplotlib.pyplot as plt


class Hysteresis_MDOF_NewMark1:
    def __init__(self, m, k0, fy, alpha, dt, device):
        self.device = device
        self.m = torch.as_tensor(m, dtype=torch.float64, device=self.device)
        self.fy = torch.as_tensor(fy, dtype=torch.float64, device=self.device)
        self.k0 = torch.as_tensor(k0, dtype=torch.float64, device=self.device)
        self.alpha = torch.as_tensor(alpha, dtype=torch.float64, device=self.device)
        self.dt = torch.as_tensor(dt, dtype=torch.float64, device=self.device)
        self.freedom = len(self.m)

        self.M = torch.diag(self.m)
        self.K0 = self.form_k_matrix(self.k0)
        self.inv_M = torch.linalg.inv(self.M)
        self.gama = 0.5
        self.beta = 0.25

        eig_val, eig_vec = torch.linalg.eig(self.inv_M @ self.K0)
        w_order = torch.argsort(torch.sqrt(torch.real(eig_val)))
        w = torch.sqrt(torch.real(eig_val[w_order]))

        A = 2 * 0.05 / (w[0] + w[1]) * torch.tensor([[w[0] * w[1]], [1]], dtype=torch.float64, device=self.device)
        self.C = A[0] * self.M + A[1] * self.K0

    def integrate(self, ag, prev_u, prev_v, prev_a):
        ag = ag.permute(0, 2, 1).to(torch.float64)
        n, num_steps = ag.size(0), ag.size(2)
        u = prev_u.permute(0, 2, 1).to(torch.float64)
        v = prev_v.permute(0, 2, 1).to(torch.float64)
        a = prev_a.permute(0, 2, 1).to(torch.float64)
        Rs = torch.zeros_like(ag, device=self.device, dtype=torch.float64)

        for i in range(1, num_steps):
            Rtrial = self.state_determine_nmdof(Rs[:, :, i-1], u[:, :, i-1], u[:, :, i])
            Rs[:, :, i] = Rtrial

        ar = (self.inv_M.unsqueeze(0) @ Rs)[:, 1:-1, :]
        e1 = (-ag - a - (self.C @ self.inv_M).unsqueeze(0)@v)[:, 1:-1, :]

        return (ar.permute(0, 2, 1).to(torch.float32),
                e1.permute(0, 2, 1).to(torch.float32))


    @staticmethod
    def form_k_matrix(k_list):
        if k_list.ndim == 2:
            n, m = k_list.shape
            K = torch.zeros((n, m, m))
            K += torch.diag_embed(k_list + torch.cat((k_list[:, 1:], torch.zeros((n, 1))), dim=1))
            k_aux = k_list[:, 1:]
            K[:, :-1, 1:] -= torch.diag_embed(k_aux)
            K[:, 1:, :-1] -= torch.diag_embed(k_aux)

        elif k_list.ndim == 1:
            if len(k_list) > 1:
                k_aux = k_list[1:]
                k_aux = torch.cat((k_aux, torch.tensor([0], dtype=k_list.dtype, device=k_list.device)))
                K = torch.diag(k_list + k_aux) - torch.diag(k_list[1:], 1) - torch.diag(k_list[1:], -1)
            else:
                K = k_list
        else:
            raise RuntimeError("Input tensor k must be either a 1D or 2D tensor")
        return K

    def state_determine_nmdof(self, last_rs, last_disp, disp):
        force = torch.zeros_like(last_rs)
        last_shear = torch.cumsum(last_rs.flip(1), dim=1).flip(1)
        last_relative_disp = torch.cat((last_disp[:, 0:1], torch.diff(last_disp, dim=1)), dim=1)
        relative_disp = torch.cat((disp[:, 0:1], torch.diff(disp, dim=1)), dim=1)

        fy1_minusb = self.fy * (1 - self.alpha)
        ksh = self.alpha * self.k0
        d_disp = relative_disp - last_relative_disp

        c1 = ksh * relative_disp
        c2 = fy1_minusb * torch.ones_like(relative_disp)
        c3 = fy1_minusb * torch.ones_like(relative_disp)
        c = last_shear + self.k0.unsqueeze(0) * d_disp
        shear = torch.clamp(c1 - c2, min=torch.min(c1 + c3, c))
        force[:, :-1] = shear[:, :-1] - shear[:, 1:]
        force[:, -1] = shear[:, -1]
        return force