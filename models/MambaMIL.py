"""
MambaMIL_SRMamba
"""
import torch
import torch.nn as nn
from mamba.mamba_ssm import SRMamba #1 as SRMamba
from mamba.mamba_ssm import BiMamba
from mamba.mamba_ssm import Mamba
# from mamba.mamba_ssm import Mamba2Simple
from mamba.mamba_ssm import MultiMamba
import torch.nn.functional as F


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class MambaMIL(nn.Module):
    def __init__(self, n_classes, dropout, act, n_features=1024, layer=2, rate=10, type="SRMamba"):
        super(MambaMIL, self).__init__()
        self._fc1 = [nn.Linear(n_features, 512)]
        if act.lower() == 'relu':
            self._fc1 += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self._fc1 += [nn.GELU()]
        if dropout:
            self._fc1 += [nn.Dropout(dropout)]

        self._fc1 = nn.Sequential(*self._fc1)
        self.norm = nn.LayerNorm(512)
        self.layers = nn.ModuleList()

        if type == "SRMamba":
            for _ in range(layer):
                self.layers.append(
                    nn.Sequential(
                        nn.LayerNorm(512),
                        SRMamba(
                            d_model=512,
                            d_state=16,
                            d_conv=4,
                            expand=2,
                        ),
                    )
                )
        # elif type == "Mamba":
        #     for _ in range(layer):
        #         self.layers.append(
        #             nn.Sequential(
        #                 nn.LayerNorm(512),
        #                 MambaSimple(
        #                     d_model=512,
        #                     d_state=16,
        #                     d_conv=4,
        #                     expand=2,
        #                 ),
        #             )
        #         )
        elif type == "MultiMamba":
            for _ in range(layer):
                self.layers.append(
                    nn.Sequential(
                        nn.LayerNorm(512),
                        MultiMamba(
                            d_model=512,
                            d_state=16,
                            d_conv=4,
                            expand=2,
                        ),
                    )
                )
        elif type == "BiMamba":
            for _ in range(layer):
                self.layers.append(
                    nn.Sequential(
                        nn.LayerNorm(512),
                        BiMamba(
                            d_model=512,
                            d_state=16,
                            d_conv=4,
                            expand=2,
                        ),
                    )
                )
        else:
            raise NotImplementedError("Mamba [{}] is not implemented".format(type))

        self.n_classes = n_classes
        self.classifier = nn.Linear(512, self.n_classes)
        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.rate = rate
        self.type = type

        self.apply(initialize_weights)

    # def forward(self, x):
    def forward(self, **kwargs):

        h = kwargs['data'].float()  # [B, n, 1024]
        # h = x.float()  # [B, n, 1024]
        B, N, C = h.size()

        h = self._fc1(h)  # [B, n, 512]

        if self.type == "SRMamba" or self.type == "MultiMamba":
            for layer in self.layers:
                h_ = h
                h = layer[0](h)  # layerNorm
                h = layer[1](h, rate=self.rate)  # SRMamba   [B, n, 512]
                h = h + h_  # [B, n, 512]
        elif self.type == "Mamba" or self.type == "BiMamba":
            for layer in self.layers:
                h_ = h
                h = layer[0](h)
                h = layer[1](h)
                h = h + h_

        h = self.norm(h)
        A = self.attention(h)  # [B, n, K]
        A = torch.transpose(A, 1, 2)
        A_1 = A
        A = F.softmax(A, dim=-1)  # [B, K, n]
        h = torch.bmm(A, h)  # [B, K, 512]
        h = h.squeeze(0)
        # ---->predict
        logits = self.classifier(h)  # [B, n_classes]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat, 'weights': A_1, 'feature':h}
        # hazards = torch.sigmoid(logits)
        # S = torch.cumprod(1 - hazards, dim=1)
        # return hazards, S
        return results_dict
