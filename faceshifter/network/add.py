import torch
from torch import nn


class ADD(nn.Module):
    def __init__(self, c_x, c_att, c_id):
        super(ADD, self).__init__()

        self.c_x = c_x

        self.h_conv = nn.Conv2d(in_channels=c_x, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

        self.att_conv1 = nn.Conv2d(in_channels=c_att, out_channels=c_x, kernel_size=1, stride=1, padding=0, bias=True)
        self.att_conv2 = nn.Conv2d(in_channels=c_att, out_channels=c_x, kernel_size=1, stride=1, padding=0, bias=True)

        self.id_fc1 = nn.Linear(c_id, c_x)
        self.id_fc2 = nn.Linear(c_id, c_x)

        self.norm = nn.InstanceNorm2d(c_x, affine=False)

    def forward(self, h, z_att, z_id):
        h_norm = self.norm(h)

        att_beta = self.att_conv1(z_att)
        att_gamma = self.att_conv2(z_att)

        id_beta = self.id_fc1(z_id)
        id_gamma = self.id_fc2(z_id)

        id_beta = id_beta.reshape(h_norm.shape[0], self.c_x, 1, 1).expand_as(h_norm)
        id_gamma = id_gamma.reshape(h_norm.shape[0], self.c_x, 1, 1).expand_as(h_norm)

        M = torch.sigmoid(self.h_conv(h_norm))
        A = att_gamma * h_norm + att_beta
        I = id_gamma * h_norm + id_beta

        return (torch.ones_like(M).to(M.device) - M) * A + M * I


def conv(c_in, c_out):
    return nn.Sequential(
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=3, stride=1, padding=1, bias=False),
    )


class ADDResBlk(nn.Module):
    def __init__(self, c_in, c_out, c_att, c_id):
        super(ADDResBlk, self).__init__()

        self.c_in = c_in
        self.c_out = c_out

        self.add1 = ADD(c_in, c_att, c_id)
        self.conv1 = conv(c_in, c_in)
        self.add2 = ADD(c_in, c_att, c_id)
        self.conv2 = conv(c_in, c_out)

        if c_in != c_out:
            self.add3 = ADD(c_in, c_att, c_id)
            self.conv3 = conv(c_in, c_out)

    def forward(self, h, z_att, z_id):
        x = self.add1(h, z_att, z_id)
        x = self.conv1(x)
        x = self.add1(x, z_att, z_id)
        x = self.conv2(x)
        if self.c_in != self.c_out:
            h = self.add3(h, z_att, z_id)
            h = self.conv3(h)

        return x + h
