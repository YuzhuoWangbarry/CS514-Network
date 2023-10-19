import numpy as np
import torch
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv

dataset = Planetoid(root='/tmp/Cora', name='Cora')
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)
    
class GAT(torch.nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.conv1 = GATConv(dataset.num_node_features, 8, heads=8, dropout=0.6)
        self.conv2 = GATConv(8*8, dataset.num_classes, heads=1, concat=False, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# test GCN
modelGCN = GCN().to(device)
# test GAT
modelGAT = GAT().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(modelGCN.parameters(), lr=0.01, weight_decay=5e-4)

modelGCN.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = modelGCN(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

modelGAT.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = modelGAT(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

modelGCN.eval()
_,pred = modelGCN(data).max(dim=1)
correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct / data.test_mask.sum().item()
print('GCN accuracy: {:.4f}'.format(acc))

modelGAT.eval()
predGAT = modelGAT(data).max(dim=1)[1]
correctGAT = float(predGAT[data.test_mask].eq(data.y[data.test_mask]).sum().item())
accGAT = correctGAT/data.test_mask.sum().item()
print('GAT accuracy: {:.4f}'.format(accGAT))
