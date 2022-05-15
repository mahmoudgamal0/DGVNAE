import os.path as osp

import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import train_test_split_edges
from torch_geometric.nn import GAE, VGAE
import torch_geometric.transforms as T
import statistics

from model import GNAE_ENC, VGNAE_ENC
from utils import save_model

MODEL = 'GNAE'

def run(args):
  DATASET = args['dataset']
  CHANNEL = args['channel']
  DEPTH = args['depth']
  LINK = args['link']
  PI = args['pi']
  TP = args['tp']
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
      model = GAE(GNAE_ENC(N_F, CHANNEL, CHANNEL, DEPTH, LINK, K=PI, alpha=TP)).to(device)
  else:
      model = VGAE(VGNAE_ENC(N_F, CHANNEL, CHANNEL, DEPTH, LINK, K=PI, alpha=TP)).to(device)

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

  auc_list, ap_list = [], []
  loss = 0

  for epoch in range(1, EPOCHS):
    loss = train()
    loss = float(loss)
    
    with torch.no_grad():
      auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
      print('Epoch: {:03d}, LOSS: {:.4f}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, loss, auc, ap))
      auc_list.append(auc*100)
      ap_list.append(ap*100)

  auc = 'AUC score: {:.2f} +/-{:.2f}'.format(statistics.mean(auc_list), statistics.stdev(auc_list))
  ap = 'AP score: {:.2f} +/- {:.2f}'.format(statistics.mean(ap_list), statistics.stdev(ap_list)) 
  print(auc)
  print(ap)
  print("------------------------")

  save_model(EPOCHS, model, optimizer, loss, f'{MODEL}:{DATASET}_{CHANNEL}_{DEPTH}_{LINK}_{PI}_{TP}', auc, ap)



DATASETS = ['Cora', 'CiteSeer', 'PubMed']
CHANNELS = [32, 64, 128, 512]
DEPTHS = [3, 5, 8]
LINKS = [1, 2, 3, 4]
PROPAGATION_ITERATION = [2, 4, 6, 8, 10]
TELEPORTAION_PROPAPILITY = [0.1, 0.15, 0.2, 0.25]

for dataset in DATASETS:
  for channel in CHANNELS:
    for depth in DEPTHS:
      for link in LINKS:
        for pi in PROPAGATION_ITERATION:
          for tp in TELEPORTAION_PROPAPILITY:
            args = {
              'dataset': dataset,
              'depth': depth,
              'channel': channel,
              'link': link,
              'pi': pi,
              'tp': tp
            }

            try:
              run(args)
            except Exception as e:
              print(e)