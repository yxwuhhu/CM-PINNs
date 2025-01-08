# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMModel1(nn.Module):
    def __init__(self, n_hidden):
        super(LSTMModel1, self).__init__()
        self.n_hidden = n_hidden
        self.lstm1 = nn.LSTM(1, n_hidden, batch_first=True)
        self.activation1 = nn.ReLU()
        self.lstm2 = nn.LSTM(n_hidden, n_hidden, batch_first=True)
        self.activation2 = nn.ReLU()
        self.N2L1 = nn.LSTM(1 + self.n_hidden, self.n_hidden, batch_first=True)
        self.N2FC3 = nn.Linear(self.n_hidden, self.n_hidden, bias=True)
        self.N2_OUT = nn.Linear(self.n_hidden, 3)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.activation1(out)
        out, _ = self.lstm2(out)
        out = self.activation2(out)

        # Net2
        N2L1_out, _ = self.N2L1(torch.cat([x, out], dim=2))
        N2FC3_out = F.relu(self.N2FC3(N2L1_out))
        uvr = self.N2_OUT(N2FC3_out)
        return uvr


class LSTMModel2(nn.Module):
    def __init__(self, n_hidden):
        super(LSTMModel2, self).__init__()
        self.n_hidden = n_hidden
        self.lstm1 = nn.LSTM(3, n_hidden, batch_first=True)
        self.activation1 = nn.ReLU()
        self.lstm2 = nn.LSTM(n_hidden, n_hidden, batch_first=True)
        self.activation2 = nn.ReLU()
        self.N2L1 = nn.LSTM(3 + self.n_hidden, self.n_hidden, batch_first=True)
        self.N2FC3 = nn.Linear(self.n_hidden, self.n_hidden, bias=True)
        self.N2_OUT = nn.Linear(self.n_hidden, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.activation1(out)
        out, _ = self.lstm2(out)
        out = self.activation2(out)

        # Net2
        N2L1_out, _ = self.N2L1(torch.cat([x, out], dim=2))
        N2FC3_out = F.relu(self.N2FC3(N2L1_out))
        g = self.N2_OUT(N2FC3_out)  # g
        return g


class LSTMModel3(nn.Module):
    def __init__(self, n_hidden):
        super(LSTMModel3, self).__init__()
        self.n_hidden = n_hidden
        self.lstm1 = nn.LSTM(2, n_hidden, batch_first=True)
        self.activation1 = nn.ReLU()
        self.lstm2 = nn.LSTM(n_hidden, n_hidden, batch_first=True)
        self.activation2 = nn.ReLU()
        self.N2L1 = nn.LSTM(2 + self.n_hidden, self.n_hidden, batch_first=True)
        self.N2FC3 = nn.Linear(self.n_hidden, self.n_hidden, bias=True)
        self.N2_OUT = nn.Linear(self.n_hidden, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.activation1(out)
        out, _ = self.lstm2(out)
        out = self.activation2(out)

        # Net2
        N2L1_out, _ = self.N2L1(torch.cat([x, out], dim=2))
        N2FC3_out = F.relu(self.N2FC3(N2L1_out))
        rdot = self.N2_OUT(N2FC3_out)  # ar
        return rdot


class PhyLSTM3(nn.Module):
    def __init__(self, n_hidden, Fin_DIF):
        super(PhyLSTM3, self).__init__()
        self.Fin_DIF = Fin_DIF
        self.Net1 = LSTMModel1(n_hidden=n_hidden)
        self.Net2 = LSTMModel2(n_hidden=n_hidden)
        self.Net3 = LSTMModel3(n_hidden=n_hidden)

    def forward(self, ag, model_type=''):
        uvr = self.Net1(x=ag)
        u = uvr[:, :, 0:1]
        udot = uvr[:, :, 1:2]
        r = uvr[:, :, 2:]
        g = self.Net2(x=uvr)
        v = self.Fin_DIF.finite_difference2(u)
        vdot = self.Fin_DIF.finite_difference2(udot)
        rdot1 = self.Fin_DIF.finite_difference2(r)
        udot_with_zero = torch.cat((torch.zeros(udot.size(0), 1, 1).cuda(), udot), dim=1)
        dlt_udot = torch.diff(udot_with_zero, dim=1)
        rdot2 = self.Net3(x=torch.cat((dlt_udot, r), dim=-1))
        if model_type == 'data':
            output = torch.cat((u, udot, vdot, g), dim=-1)
        elif model_type == 'phy':
            output = torch.cat((udot, v, vdot, g, rdot1, rdot2), dim=-1)
        else:
            output = torch.cat((u, v, vdot, g), dim=-1)
        return output




