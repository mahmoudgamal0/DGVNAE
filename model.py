
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class GNAE_ENC(torch.nn.Module):
		def __init__(self, in_dim, hidden_dim, out_dim, depth=3, link_len=2):
			super(GNAE_ENC, self).__init__()

			self.depth = depth
			self.link_len = link_len
			assert self.depth > self.link_len, "Link length cannot be bigger than depth"

			self.convs = torch.nn.ModuleList()
			self.convs.append(GCNConv(in_dim, hidden_dim))
			for _ in range(1, depth):
				self.convs.append(GCNConv(hidden_dim, hidden_dim))
			
			self.convx = GCNConv(hidden_dim, out_dim)
		
		def forward(self, x, edge_index):
			out = F.relu(self.convs[0](x, edge_index))
			res, res_ind = out, 0
			for i in range(1, self.depth-1):
				if i == res_ind + self.link_len:
					out = out + res
				out = F.normalize(self.convs[i](out, edge_index),p=2,dim=1) * 1.5
				if i == res_ind + self.link_len:
					res = out
					res_ind = i

			return self.convx(out, edge_index)

class VGNAE_ENC(torch.nn.Module):
		def __init__(self, in_dim, hidden_dim, out_dim, depth=3, link_len=2):
			super(VGNAE_ENC, self).__init__()

			self.depth = depth
			self.link_len = link_len
			assert self.depth > self.link_len, "Link length cannot be bigger than depth"

			self.convs = torch.nn.ModuleList()
			self.convs.append(GCNConv(in_dim, hidden_dim))
			for _ in range(1, depth):
				self.convs.append(GCNConv(hidden_dim, hidden_dim))

			self.conv_mu = GCNConv(hidden_dim, out_dim)
			self.conv_logstd = GCNConv(hidden_dim, out_dim)

		def forward(self, x, edge_index):
			out = F.relu(self.convs[0](x, edge_index))
			res, res_ind = out, 0
			for i in range(1, self.depth-1):
				if i == res_ind + self.link_len:
					out = out + res
				out = F.relu(self.convs[i](out, edge_index))
				if i == res_ind + self.link_len:
					res = out
					res_ind = i

			out = self.convs[-1](out, edge_index)

			return self.conv_mu(out, edge_index), self.conv_logstd(out, edge_index)


class InnerProductDecoder(nn.Module):
  """Decoder for using inner product for prediction."""

  def __init__(self, dropout, act=torch.sigmoid):
    super(InnerProductDecoder, self).__init__()
    self.dropout = dropout
    self.act = act

  def forward(self, z):
    z = F.dropout(z, self.dropout, training=self.training)
    adj = self.act(torch.mm(z, z.t()))
    return adj
