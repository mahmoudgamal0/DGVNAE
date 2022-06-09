import os.path as osp
import matplotlib
matplotlib.use('TkAgg')
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import train_test_split_edges
from torch_geometric.nn import GAE, VGAE
import torch_geometric.transforms as T

from model import GNAE_ENC, VGNAE_ENC
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


DATASET='Cora'
CHANNEL=128
DEPTH=3
LINK=2
PI=4
TP=0.15

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', DATASET)

dataset = Planetoid(path, DATASET, 'public')

data = dataset[0]

N_F = int(data.num_features)
model = GAE(GNAE_ENC(N_F, CHANNEL, CHANNEL, DEPTH, LINK, K=PI, alpha=TP)).to(device)
model.load_state_dict(torch.load("outputs/GNAE:Cora_128_3_2_4_0.15.pth")['model_state_dict'])
model.eval()

x, y, edge_index = data.x.to(device), data.y, data.edge_index.to(device)



res = model.encode(x, edge_index)

tnse_model = TSNE(n_components=3, random_state=42)

transformed = tnse_model.fit_transform(res.detach().cpu())

# import ipdb; ipdb.set_trace()
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

labels = np.unique(y.cpu().numpy())
scatter = ax.scatter(transformed[:,0], transformed[:,1], transformed[:,2], marker='o', c=y)
handles, _ = scatter.legend_elements(prop='colors')
plt.legend(handles, labels)
plt.show()
