import torch
import torch.nn.functional as func
import torch_geometric
from torch_geometric.nn import GATv2Conv
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Amazon
from torch_geometric.transforms import RandomNodeSplit
import numpy as np
import networkx as nx
import csv
# import matplotlib
# matplotlib.use('TKAgg')
import matplotlib.pyplot as plt


class GATv2(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden=8, heads=8):
        super(GATv2, self).__init__()
        self.conv1 = GATv2Conv(num_node_features, hidden, heads=heads)
        self.conv2 = GATv2Conv(heads * hidden, hidden, heads=heads)
        self.conv3 = GATv2Conv(heads * hidden, hidden, heads=heads)
        self.conv4 = GATv2Conv(heads * hidden, hidden, heads=heads)
        self.conv5 = GATv2Conv(heads * hidden, num_classes, heads=heads, concat=False)

        # Use torch.nn.Linear for residual connections
        self.res1 = torch.nn.Linear(num_node_features, heads * hidden)
        self.res2 = torch.nn.Linear(heads * hidden, heads * hidden)
        self.res3 = torch.nn.Linear(heads * hidden, heads * hidden)
        self.res4 = torch.nn.Linear(heads * hidden, heads * hidden)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Apply GATv2 layers with residual connections
        x1 = self.conv1(x, edge_index)
        x = func.leaky_relu(x1 + self.res1(x), negative_slope=0.2)
        x = func.dropout(x, p=0.3, training=self.training)

        x2 = self.conv2(x, edge_index)
        x = func.leaky_relu(x2 + self.res2(x), negative_slope=0.2)
        x = func.dropout(x, p=0.3, training=self.training)

        x3 = self.conv3(x, edge_index)
        x = func.leaky_relu(x3 + self.res3(x), negative_slope=0.2)
        x = func.dropout(x, p=0.3, training=self.training)

        x4 = self.conv4(x, edge_index)
        x = func.leaky_relu(x4 + self.res4(x), negative_slope=0.2)
        x = func.dropout(x, p=0.3, training=self.training)

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
    with open("CSV_files/GATv2_Residual.csv", 'w', newline='') as csvfile:
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
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            
            optimizer.step()

            train_acc = test(data)
            test_acc = test(data, train=False)

            train_accurate.append(train_acc)
            test_accurate.append(test_acc)
            writer.writerow([epoch, loss_values, train_acc, test_acc])
            # print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, Test Acc: {:.5f}'.
            #         format(epoch, loss, train_acc, test_acc))

if __name__ == "__main__":
    # Load the Amazon dataset
    dataset = Amazon(root='/tmp/Amazon', name='Computers')

    # Apply the transform to the single Data object in the dataset
    transform = RandomNodeSplit(split='random', num_train_per_class=20, num_val=500, num_test=1000)
    data = transform(dataset[0])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Normalization
    data.x = (data.x - data.x.min(0, keepdim=True)[0]) / (data.x.max(0, keepdim=True)[0] - data.x.min(0, keepdim=True)[0])

    # Determine the number of classes
    num_classes = len(torch.unique(data.y))
    num_node_features = data.x.size(1)

    # Corrected model initialization
    model = GATv2(num_node_features, num_classes).to(device) # Pass the number of node features and number of classes

    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
   # criterion = torch.nn.CrossEntropyLoss()


    train(data)