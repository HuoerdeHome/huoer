import torch
from torch import nn

class CooGuidance(nn.Module):

    def __init__(self, graph_feat_size):
        super(CooGuidance, self).__init__()
        # co-guided networks
        self.w_p_z = nn.Linear(graph_feat_size, graph_feat_size, bias=True)
        self.w_p_r = nn.Linear(graph_feat_size, graph_feat_size, bias=True)
        self.w_p = nn.Linear(graph_feat_size, graph_feat_size, bias=True)

        self.u_i_z = nn.Linear(graph_feat_size, graph_feat_size, bias=True)
        self.u_i_r = nn.Linear(graph_feat_size, graph_feat_size, bias=True)
        self.u_i = nn.Linear(graph_feat_size, graph_feat_size, bias=True)

        # gate5 & gate6
        self.w_pi_1 = nn.Linear(graph_feat_size, graph_feat_size, bias=True)
        self.w_pi_2 = nn.Linear(graph_feat_size, graph_feat_size, bias=True)
        self.w_c_z = nn.Linear(graph_feat_size, graph_feat_size, bias=True)
        self.u_j_z = nn.Linear(graph_feat_size, graph_feat_size, bias=True)
        self.w_c_r = nn.Linear(graph_feat_size, graph_feat_size, bias=True)
        self.u_j_r = nn.Linear(graph_feat_size, graph_feat_size, bias=True)
        self.w_p = nn.Linear(graph_feat_size, graph_feat_size, bias=True)
        self.u_p = nn.Linear(graph_feat_size, graph_feat_size, bias=True)
        self.w_i = nn.Linear(graph_feat_size, graph_feat_size, bias=True)
        self.u_i = nn.Linear(graph_feat_size, graph_feat_size, bias=True)

    def forward(self, HG1, HG2):
        HG1 = HG1.float()
        HG2 = HG2.float()
        m_c = torch.tanh(self.w_pi_1(HG1 * HG2))
        m_j = torch.tanh(self.w_pi_2(HG1 + HG2))

        r_i = torch.sigmoid(self.w_c_z(m_c) + self.u_j_z(m_j))
        r_p = torch.sigmoid(self.w_c_r(m_c) + self.u_j_r(m_j))

        m_p = torch.tanh(self.w_p(HG1 * r_p) + self.u_p((1 - r_p) * HG2))
        m_i = torch.tanh(self.w_i(HG2 * r_i) + self.u_i((1 - r_i) * HG1))

        # enriching the semantics of price and interest preferences
        HG1_C = (HG1 + m_i) * m_p
        HG2_C = (HG2 + m_p) * m_i

        return HG1_C, HG2_C