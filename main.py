import os.path as osp

import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import train_test_split_edges
from torch_geometric.nn import GAE, VGAE
import torch_geometric.transforms as T
import statistics

from model import GNAE_ENC, VGNAE_ENC

DATASET = 'Cora'
MODEL = 'GNAE'
CHANNELS = 64
EPOCHS = 300

path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', DATASET)
dataset = Planetoid(path, DATASET, 'public')

data = dataset[0]

# Why this??
data = T.NormalizeFeatures()(data)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = train_test_split_edges(data.to(device), val_ratio=0.2, test_ratio=0.2)

N = int(data.num_nodes)
N_F = int(data.num_features)


# FIX Models
if MODEL == 'GNAE':   
    model = GAE(GNAE_ENC(N_F, CHANNELS, CHANNELS, 8, 3)).to(device)
else:
    model = VGAE(VGNAE_ENC(N_F, CHANNELS, CHANNELS, 8, 2)).to(device)

data.train_mask = data.val_mask = data.test_mask = data.y = None
x, train_pos_edge_index = data.x.to(device), data.train_pos_edge_index.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train():
  model.train()
  optimizer.zero_grad()
  z  = model.encode(x, train_pos_edge_index)
  loss = model.recon_loss(z, train_pos_edge_index)
  if MODEL == 'VGNAE':
      loss = loss + (1 / data.num_nodes) * model.kl_loss()
  loss.backward()
  optimizer.step()
  return loss

def test(pos_edge_index, neg_edge_index):
  model.eval()
  with torch.no_grad():
      z = model.encode(x, train_pos_edge_index)
  return model.test(z, pos_edge_index, neg_edge_index)

print(model)
# import ipdb; ipdb.set_trace()
auc_list, ap_list = [], []
for epoch in range(1, EPOCHS):
  loss = train()
  loss = float(loss)
  
  with torch.no_grad():
    test_pos, test_neg = data.test_pos_edge_index, data.test_neg_edge_index
    auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
    print('Epoch: {:03d}, LOSS: {:.4f}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, loss, auc, ap))
    auc_list.append(auc*100)
    ap_list.append(ap*100)

       

print('AUC score: {:.2f} +/-{:.2f}'.format(statistics.mean(auc_list), statistics.stdev(auc_list)))
print('AP score: {:.2f} +/- {:.2f}'.format(statistics.mean(ap_list), statistics.stdev(ap_list)))
print("------------------------")