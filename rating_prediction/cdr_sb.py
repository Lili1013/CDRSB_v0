import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from attention import Attention

class CDR_SB(nn.Module):
    def __init__(self, rep_u,rep_v,mi_net,device):
        super(CDR_SB, self).__init__()
        self.rep_u = rep_u
        self.rep_v = rep_v
        self.embed_dim = rep_u.dec_embed_dim
        self.mi_net = mi_net
        self.device = device
        # self.tau = tau
        self.att = Attention(rep_u.embed_dim,self.embed_dim)

        self.w_uv_1 = nn.Linear(self.embed_dim*2, self.embed_dim)
        nn.init.xavier_uniform_(self.w_uv_1.weight)
        self.w_uv_2 = nn.Linear(self.embed_dim, 1)
        self.uv_bn = nn.BatchNorm1d(self.embed_dim)

        self.criterion = nn.MSELoss()

    def forward(self,nodes_u,nodes_v):
        embed_u_matrix = torch.empty(len(nodes_u), self.embed_dim, dtype=torch.float).to(self.device)
        embed_u_z, embed_u_c, embed_u = self.rep_u(nodes_u)
        for i in range(len(embed_u)):
            aat_u = self.att(embed_u_z[i], embed_u_c[i], embed_u[i])
            # print(aat_u)
            u = aat_u[0] * embed_u_z[i] + aat_u[1] * embed_u_c[i]
            embed_u_matrix[i] = u
        embed_v_matrix = torch.empty(len(nodes_v), self.embed_dim, dtype=torch.float).to(self.device)
        embed_v_z, embed_v_c, embed_v = self.rep_v(nodes_v)
        for i in range(len(embed_v)):
            aat_v = self.att(embed_v_z[i], embed_v_c[i], embed_v[i])
            # print(aat_v)
            v = aat_v[0] * embed_v_z[i] + aat_v[1] * embed_v_c[i]
            embed_v_matrix[i] = v
        uv = torch.cat([embed_u_matrix, embed_v_matrix], dim=1)
        uv = F.dropout(F.relu(self.uv_bn(self.w_uv_1(uv))),training=self.training)
        uv_score = self.w_uv_2(uv)
        return uv_score.squeeze()
    
    def loss(self,nodes_u,nodes_v,scores,labels_list):
        # scores = self.forward(nodes_u, nodes_v)
        # cl_total_loss = self.calc_ssl_loss_strategy()
        mi_total_loss = self.calculate_mi_loss(nodes_u,nodes_v)

        prediction_loss = self.criterion(scores, labels_list)

        return prediction_loss,mi_total_loss

    def calculate_mi_loss(self,nodes_u,nodes_v):
        loglikeli_zc_loss_u, bound_zc_loss_u = self.mi_net(nodes_u, name='zc',u=True)
        # loglikeli_za_loss_u, bound_za_loss_u = self.mi_net(nodes_u, name='za',u=True)
        # loglikeli_ca_loss_u, bound_ca_loss_u = self.mi_net(nodes_u, name='ca',u=True)
        oglikeli_zc_loss_v, bound_zc_loss_v = self.mi_net(nodes_v, name='zc',u=False)
        # loglikeli_za_loss_v, bound_za_loss_v = self.mi_net(nodes_v, name='za',u=False)
        # loglikeli_ca_loss_v, bound_ca_loss_v = self.mi_net(nodes_v, name='ca',u=False)
        mi_total_loss = loglikeli_zc_loss_u+oglikeli_zc_loss_v\
                  +bound_zc_loss_u+\
                  bound_zc_loss_v
        return mi_total_loss




