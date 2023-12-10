import torch
import torch.nn.functional as func
import torch_geometric
from torch_geometric.nn import GATv2Conv
from torch_geometric.datasets import Planetoid
import numpy as np
import networkx as nx
import csv
# import matplotlib
# matplotlib.use('TKAgg')
import matplotlib.pyplot as plt


class GATv2(torch.nn.Module):
    def __init__(self, dataset, hidden=8, heads=8):
        super(GATv2, self).__init__()
        self.conv1 = GATv2Conv(dataset.num_node_features, hidden, heads=heads)
        self.conv2 = GATv2Conv(heads * hidden, hidden, heads=heads)
        self.conv3 = GATv2Conv(heads * hidden, hidden, heads=heads)
        self.conv4 = GATv2Conv(heads * hidden, hidden, heads=heads)
        self.conv5 = GATv2Conv(heads * hidden, dataset.num_classes, heads=heads, concat=False)

        # Use torch.nn.Linear for residual connections
        self.res1 = torch.nn.Linear(dataset.num_node_features, heads * hidden)
        self.res2 = torch.nn.Linear(heads * hidden, heads * hidden)
        self.res3 = torch.nn.Linear(heads * hidden, heads * hidden)
        self.res4 = torch.nn.Linear(heads * hidden, heads * hidden)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Apply GATv2 layers with residual connections
        x1 = self.conv1(x, edge_index)
        x = func.leaky_relu(x1 + self.res1(x), negative_slope=0.2)
        x = func.dropout(x, training=self.training)

        x2 = self.conv2(x, edge_index)
        x = func.leaky_relu(x2 + self.res2(x), negative_slope=0.2)
        x = func.dropout(x, training=self.training)

        x3 = self.conv3(x, edge_index)
        x = func.leaky_relu(x3 + self.res3(x), negative_slope=0.2)
        x = func.dropout(x, training=self.training)

        x4 = self.conv4(x, edge_index)
        x = func.leaky_relu(x4 + self.res4(x), negative_slope=0.2)
        x = func.dropout(x, training=self.training)

        # The last layer does not need a residual connection
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
    with open("CSV_files/GATv2_Cora.csv", 'w', newline='') as csvfile:
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
    dataset = Planetoid(root='/tmp/Cora', name='Cora')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset.x = (dataset.x - dataset.x.min(0, keepdim=True)[0]) / (dataset.x.max(0, keepdim=True)[0] - dataset.x.min(0, keepdim=True)[0])

    model = GATv2(dataset).to(device) #Two layers training
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    # criterion = torch.nn.CrossEntropyLoss()

    train(data)