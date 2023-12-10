import torch
import numpy as np
import networkx as nx
import csv
import torch.nn.functional as func
import torch_geometric
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Reddit

# import matplotlib
# matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

# Train two layers
class GCN_two(torch.nn.Module):
    def __init__(self, dataset):
        super(GCN_two, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = func.relu(self.conv1(x, edge_index))
        x = func.dropout(x, training=self.training)
        
        x = self.conv2(x, edge_index)
        
        return func.log_softmax(x, dim=1)
# Train five layers
class GCN_five(torch.nn.Module):
    def __init__(self, dataset):
        super(GCN_five, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16,16)
        self.conv3 = GCNConv(16,16)
        self.conv4 = GCNConv(16,16)
        self.conv5 = GCNConv(16, dataset.num_classes)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = func.relu(self.conv1(x, edge_index))
        x = func.dropout(x, training=self.training)
        
        x = func.relu(self.conv2(x, edge_index))
        x = func.dropout(x, training=self.training)
        
        x = func.relu(self.conv3(x, edge_index))
        x = func.dropout(x, training=self.training)

        x = func.relu(self.conv4(x, edge_index))
        x = func.dropout(x, training=self.training)
        
        x = self.conv5(x, edge_index)
        
        return func.log_softmax(x, dim=1)
    
# Train ten layers
class GCN_ten(torch.nn.Module):
    def __init__(self, dataset):
        super(GCN_ten, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16,16)
        self.conv3 = GCNConv(16,16)
        self.conv4 = GCNConv(16,16)
        self.conv5 = GCNConv(16,16)
        self.conv6 = GCNConv(16,16)
        self.conv7 = GCNConv(16,16)
        self.conv8 = GCNConv(16,16)
        self.conv9 = GCNConv(16,16)
        self.conv10 = GCNConv(16, dataset.num_classes)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = func.relu(self.conv1(x, edge_index))
        x = func.dropout(x, training=self.training)
        
        x = func.relu(self.conv2(x, edge_index))
        x = func.dropout(x, training=self.training)
        
        x = func.relu(self.conv3(x, edge_index))
        x = func.dropout(x, training=self.training)

        x = func.relu(self.conv4(x, edge_index))
        x = func.dropout(x, training=self.training)

        x = func.relu(self.conv5(x, edge_index))
        x = func.dropout(x, training=self.training)

        x = func.relu(self.conv6(x, edge_index))
        x = func.dropout(x, training=self.training)

        x = func.relu(self.conv7(x, edge_index))
        x = func.dropout(x, training=self.training)

        x = func.relu(self.conv8(x, edge_index))
        x = func.dropout(x, training=self.training)
        
        x = func.relu(self.conv9(x, edge_index))
        x = func.dropout(x, training=self.training)

        x = self.conv10(x, edge_index)
        
        return func.log_softmax(x, dim=1)
    
def test(data, train=True):
    model.eval()

    correct = 0
    pred = model(data).max(dim = 1)[1]

    if train:
        correct += pred[data.train_mask].eq(data.y[data.train_mask]).sum().item()
        return correct / len(data.y[data.train_mask])
    else:
        correct += pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
        return correct / (len(data.y[data.test_mask]))

def train(data):
    train_accurate, test_accurate, losses = list(), list(), list()
    with open("GCN_ACC.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Epoch', 'Loss', 'Train Accuracy', 'Test Accuracy'])

        for epoch in range(100):
            model.train()
            optimizer.zero_grad()
            out = model(data)
            loss = func.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss_values = loss.item()
            losses.append(loss_values)
            loss.backward()
            optimizer.step()

            train_acc = test(data)
            test_acc = test(data, train=False)

            train_accurate.append(train_acc)
            test_accurate.append(test_acc)
            writer.writerow([epoch, loss_values, train_acc, test_acc])
            # print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, Test Acc: {:.5f}'. 
            #     format(epoch, loss, train_acc, test_acc))
        
    
        # for epoch, (losses, train_acc, test_acc) in enumerate(zip(losses, train_accurate, test_accurate)):
        #     writer.writerow([epoch, loss_values, train_acc, test_acc])
        
if __name__ == "__main__":
    dataset = Planetoid(root='/tmp/Cora', name='Cora')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = GCN_two(dataset).to(device) #Using two layers model for training
    model = GCN_five(dataset).to(device) #Using five layers model for training
    # model = GCN_ten(dataset).to(device) #Using ten layers model for training
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    train(data)
