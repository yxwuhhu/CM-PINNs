# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMModel_uvr(nn.Module):
    def __init__(self, n_hidden):
        super(LSTMModel_uvr, self).__init__()
        self.n_hidden = n_hidden
        self.lstm1 = nn.LSTM(1, self.n_hidden, batch_first=True)
        self.activation1 = nn.ReLU()
        self.lstm2 = nn.LSTM(self.n_hidden, self.n_hidden, batch_first=True)
        self.activation2 = nn.ReLU()
        self.fc1 = nn.Linear(self.n_hidden, 3*3)
        self.Res_ag = nn.Sequential(nn.Linear(1, self.n_hidden, bias=True), nn.ReLU(),
                                    nn.Linear(self.n_hidden, self.n_hidden, bias=True), nn.ReLU())
        self.lstm3 = nn.LSTM(3 * 3 + self.n_hidden, self.n_hidden, batch_first=True)
        self.fc2 = nn.Linear(self.n_hidden, 3 * 3)

    def forward(self, ag):
        out, _ = self.lstm1(ag)
        out = self.activation1(out)
        out, _ = self.lstm2(out)
        out = self.activation2(out)
        out = self.fc1(out)  # u, v, r
        Res_ag = self.Res_ag(ag)
        out, _ = self.lstm3(torch.cat([Res_ag, out], dim=2))
        uvr = self.fc2(out)
        return uvr


class LSTMModel_g(nn.Module):
    def __init__(self, n_hidden):
        super(LSTMModel_g, self).__init__()
        self.n_hidden = n_hidden
        self.lstm1 = nn.LSTM(3*3, self.n_hidden, batch_first=True)
        self.activation1 = nn.ReLU()
        self.lstm2 = nn.LSTM(self.n_hidden, self.n_hidden, batch_first=True)
        self.activation2 = nn.ReLU()
        self.fc1 = nn.Linear(self.n_hidden, 1*3)
        self.Res_uvr = nn.Sequential(nn.Linear(3 * 3, self.n_hidden, bias=True), nn.ReLU(),
                                     nn.Linear(self.n_hidden, self.n_hidden, bias=True), nn.ReLU())
        self.lstm3 = nn.LSTM(1*3 + self.n_hidden, self.n_hidden, batch_first=True)
        self.activation3 = nn.ReLU()
        self.fc2 = nn.Linear(self.n_hidden, 1*3)

    def forward(self, uvr):
        out, _ = self.lstm1(uvr)
        out = self.activation1(out)
        out, _ = self.lstm2(out)
        out = self.activation2(out)
        out = self.fc1(out)  # g
        Res_uvr = self.Res_uvr(uvr)
        out, _ = self.lstm3(torch.cat([Res_uvr, out], dim=2))
        out = self.activation3(out)
        out = self.fc2(out)  # g
        return out


class LSTMModel_rdot(nn.Module):
    def __init__(self, n_hidden):
        self.n_hidden = n_hidden
        super(LSTMModel_rdot, self).__init__()
        self.lstm1 = nn.LSTM(2*3, self.n_hidden, batch_first=True)
        self.activation1 = nn.ReLU()
        self.lstm2 = nn.LSTM(self.n_hidden, self.n_hidden, batch_first=True)
        self.activation2 = nn.ReLU()
        self.fc1 = nn.Linear(self.n_hidden, 1*3)
        self.Res_dvr = nn.Sequential(nn.Linear(2 * 3, self.n_hidden, bias=True), nn.ReLU(),
                                     nn.Linear(self.n_hidden, self.n_hidden, bias=True), nn.ReLU())
        self.lstm3 = nn.LSTM(1*3 + self.n_hidden, self.n_hidden, batch_first=True)
        self.activation3 = nn.ReLU()
        self.fc2 = nn.Linear(self.n_hidden, 1*3)

    def forward(self, dvr):
        out, _ = self.lstm1(dvr)  # x为dltudot, rdot1
        out = self.activation1(out)
        out, _ = self.lstm2(out)
        out = self.activation2(out)
        out = self.fc1(out)  # g
        Res_dvr = self.Res_dvr(dvr)
        out, _ = self.lstm3(torch.cat([Res_dvr, out], dim=2))
        out = self.activation3(out)
        out = self.fc2(out)  # rdot2
        return out


class PhyLSTM3(nn.Module):
    def __init__(self, n_hidden, Fin_DIF):
        super(PhyLSTM3, self).__init__()
        self.Fin_DIF = Fin_DIF
        self.Net_uvr = LSTMModel_uvr(n_hidden=n_hidden)
        self.Net_g = LSTMModel_g(n_hidden=n_hidden)
        self.Net_rdot = LSTMModel_rdot(n_hidden=n_hidden)

    def forward(self, ag, model_type=''):
        uvr = self.Net_uvr(ag=ag)
        g = self.Net_g(uvr=uvr)
        u = uvr[:, :, 0:1*3]
        udot = uvr[:, :, 1*3:2*3]
        r = uvr[:, :, 2*3:]
        v = self.Fin_DIF.finite_difference2(u)
        vdot = self.Fin_DIF.finite_difference2(udot)
        rdot1 = self.Fin_DIF.finite_difference2(r)
        udot_with_zero = torch.cat((torch.zeros(udot.size(0), 1, 1*3).cuda(), udot), dim=1)
        dlt_udot = torch.diff(udot_with_zero, dim=1)
        rdot2 = self.Net_rdot(dvr=torch.cat((dlt_udot, r), dim=-1))
        if model_type == 'data':
            output = torch.cat((u, udot, vdot, g), dim=-1)
        elif model_type == 'phy':
            output = torch.cat((u, udot, v, vdot, rdot1, rdot2, g), dim=-1)
        else:
            output = torch.cat((u, v, vdot, g), dim=-1)
        return output
