import sys
import os.path as osp

import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GAE, VGAE
from model import GNAE_ENC, VGNAE_ENC

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def run():
  args = sys.argv
  assert len(args) >= 2, "No model name provided"
  
  model_name = args[1]
  if '.pth' in model_name: 
    model_name = model_name[:-4]
  
  MODEL, PARAMS = model_name.split(':')
  DATASET, CHANNEL, DEPTH, LINK, PI, TP = PARAMS.split("_")
  CHANNEL, DEPTH, LINK, PI, TP = int(CHANNEL), int(DEPTH), int(LINK), int(PI), float(TP)


  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', DATASET)

  dataset = Planetoid(path, DATASET, 'public')

  data = dataset[0]
  N_F = int(data.num_features)

  if MODEL == 'GNAE':   
    model = GAE(GNAE_ENC(N_F, CHANNEL, CHANNEL, DEPTH, LINK, K=PI, alpha=TP)).to(device)
  else:
    model = VGAE(VGNAE_ENC(N_F, CHANNEL, CHANNEL, DEPTH, LINK, K=PI, alpha=TP)).to(device)

  model.load_state_dict(torch.load(f'best/{model_name}.pth')['model_state_dict'])
  model.eval()

  x, y, edge_index = data.x.to(device), data.y, data.edge_index.to(device)
  out = model.encode(x, edge_index).detach().cpu()

  tnse_3d_model = TSNE(n_components=3, random_state=42)
  tnse_2d_model = TSNE(n_components=2, random_state=42)

  out_3d = tnse_3d_model.fit_transform(out)
  out_2d = tnse_2d_model.fit_transform(out)

  fig = plt.figure(figsize=plt.figaspect(2.))
  fig.suptitle(model_name)
  
  ax = fig.add_subplot(2, 1, 1)
  ax.scatter(out_2d[:,0], out_2d[:,1], c=y)

  ax = fig.add_subplot(2, 1, 2, projection='3d')
  ax.set_xlabel('X Label')
  ax.set_ylabel('Y Label')
  ax.set_zlabel('Z Label')
  labels = np.unique(y.cpu().numpy())
  scatter = ax.scatter(out_3d[:,0], out_3d[:,1], out_3d[:,2], marker='o', c=y)
  handles, _ = scatter.legend_elements(prop='colors')
  plt.legend(handles, labels)
  plt.show()

if __name__ == '__main__':
  run()
