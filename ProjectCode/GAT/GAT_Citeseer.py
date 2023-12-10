import torch
import torch.nn.functional as func
import torch_geometric
from torch_geometric.nn import GATConv
from torch_geometric.datasets import Planetoid
import numpy as np
import networkx as nx
import csv
# import matplotlib
# matplotlib.use('TKAgg')
import matplotlib.pyplot as plt


class GAT(torch.nn.Module):
    def __init__(self, dataset, hidden=8, heads=8):
        super(GAT, self).__init__()
        self.conv1 = GATConv(dataset.num_node_features, hidden, heads=heads)
        self.conv2 = GATConv(heads * hidden, hidden, heads=heads)
        self.conv3 = GATConv(heads * hidden, hidden, heads=heads)
        self.conv4 = GATConv(heads * hidden, hidden, heads=heads)
        self.conv5 = GATConv(heads * hidden, dataset.num_classes, heads=heads, concat=False)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = func.leaky_relu(x, negative_slope=0.2)
        x = func.dropout(x, training=self.training)

        x = self.conv2(x, edge_index)
        x = func.leaky_relu(x, negative_slope=0.2)
        x = func.dropout(x, training=self.training)

        x = self.conv3(x, edge_index)
        x = func.leaky_relu(x, negative_slope=0.2)
        x = func.dropout(x, training=self.training)

        x = self.conv4(x, edge_index)
        x = func.leaky_relu(x, negative_slope=0.2)
        x = func.dropout(x, training=self.training)

        x = self.conv5(x, edge_index)
        
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
    losses, train_accurate, test_accurate = list(), list(), list()
    with open("CSV_files/GAT_Citeseer.csv", 'w', newline='') as csvfile:
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
            #         format(epoch, loss, train_acc, test_acc))

if __name__ == "__main__":
    dataset = Planetoid(root='/tmp/Citeseer', name='Citeseer')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GAT(dataset).to(device) #Two layers training
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    # criterion = torch.nn.CrossEntropyLoss()

    train(data)