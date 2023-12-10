import torch
import torch.nn.functional as func
import torch_geometric
from torch_geometric.nn import GATv2Conv
from torch_geometric.datasets import Planetoid
from sklearn.preprocessing import StandardScaler
import numpy as np
import random
import csv
import copy

# Set a fixed random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

class GATv2_ten(torch.nn.Module):
    def __init__(self, dataset, hidden=8, heads=4):
        super(GATv2_ten, self).__init__()
        self.conv1 = GATv2Conv(dataset.num_node_features, hidden, heads=heads)
        self.conv2 = GATv2Conv(heads * hidden, hidden, heads=heads)
        self.conv3 = GATv2Conv(heads * hidden, hidden, heads=heads)
        self.conv4 = GATv2Conv(heads * hidden, hidden, heads=heads)
        self.conv5 = GATv2Conv(heads * hidden, hidden, heads=heads)
        self.conv6 = GATv2Conv(heads * hidden, hidden, heads=heads)
        self.conv7 = GATv2Conv(heads * hidden, hidden, heads=heads)
        self.conv8 = GATv2Conv(heads * hidden, hidden, heads=heads)
        self.conv9 = GATv2Conv(heads * hidden, hidden, heads=heads)
        self.conv10 = GATv2Conv(heads * hidden, dataset.num_classes, heads=heads, concat=False)

        #residual connection implementation
        self.residual_lin1 = torch.nn.Linear(dataset.num_node_features, heads * hidden)
        self.residual_lin2 = torch.nn.Linear(heads * hidden, heads * hidden)
        self.residual_lin3 = torch.nn.Linear(heads * hidden, heads * hidden)
        self.residual_lin4 = torch.nn.Linear(heads * hidden, heads * hidden)
        self.residual_lin5 = torch.nn.Linear(heads * hidden, heads * hidden)
        self.residual_lin6 = torch.nn.Linear(heads * hidden, heads * hidden)
        self.residual_lin7 = torch.nn.Linear(heads * hidden, heads * hidden)
        self.residual_lin8 = torch.nn.Linear(heads * hidden, heads * hidden)
        self.residual_lin9 = torch.nn.Linear(heads * hidden, heads * hidden)
        self.residual_lin10 = torch.nn.Linear(heads * hidden, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # identity = self.residual_lin1(x)
        x = self.conv1(x, edge_index)
        x = func.leaky_relu(x, negative_slope=0.2)
        x = func.dropout(x, training=self.training)

        for i in range(2, 10):
            # identity = getattr(self, f'residual_lin{i}')(x)
            x = getattr(self, f'conv{i}')(x, edge_index)
            x = func.leaky_relu(x, negative_slope=0.2)
            x = func.dropout(x, p=0.5, training=self.training)

        # Last layer (without dropout)
        x = self.conv10(x, edge_index)
        return func.log_softmax(x, dim=1)
        
        # x = self.conv1(x, edge_index)
        # x = func.leaky_relu(x, negative_slope=0.2)
        # x = func.dropout(x, training=self.training)

        # x = self.conv2(x, edge_index)
        # x = func.leaky_relu(x, negative_slope=0.2)
        # x = func.dropout(x, training=self.training)

        # x = self.conv3(x, edge_index)
        # x = func.leaky_relu(x, negative_slope=0.2)
        # x = func.dropout(x, training=self.training)

        # x = self.conv4(x, edge_index)
        # x = func.leaky_relu(x, negative_slope=0.2)
        # x = func.dropout(x, training=self.training)

        # x = self.conv5(x, edge_index)
        # x = func.leaky_relu(x, negative_slope=0.2)
        # x = func.dropout(x, training=self.training)

        # x = self.conv6(x, edge_index)
        # x = func.leaky_relu(x, negative_slope=0.2)
        # x = func.dropout(x, training=self.training)

        # x = self.conv7(x, edge_index)
        # x = func.leaky_relu(x, negative_slope=0.2)
        # x = func.dropout(x, training=self.training)

        # x = self.conv8(x, edge_index)
        # x = func.leaky_relu(x, negative_slope=0.2)
        # x = func.dropout(x, training=self.training)

        # x = self.conv9(x, edge_index)
        # x = func.leaky_relu(x, negative_slope=0.2)
        # x = func.dropout(x, training=self.training)

        # x = self.conv10(x, edge_index)
        # return func.log_softmax(x, dim=1)


def evaluate(data, mask):
    model.eval()
    with torch.no_grad():
        logits = model(data)
        pred = logits.max(1)[1]
        correct = pred[mask].eq(data.y[mask]).sum().item()
        acc = correct / int(mask.sum())
    return acc
def normalize_features(data):
    scaler = StandardScaler()
    data.x = scaler.fit_transform(data.x)
    return data

# def train(data):
#     losses, train_accurate, test_accurate = list(), list(), list()
#     with open("GATv2_ACC.csv", 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(['Epoch', 'Loss', 'Train Accuracy', 'Test Accuracy'])

#         for epoch in range(150):
#             model.train()
#             optimizer.zero_grad()
#             out = model(data)
#             loss = func.nll_loss(out[data.train_mask], data.y[data.train_mask])
#             loss.backward()
#             optimizer.step()

#             train_acc = evaluate(data, data.train_mask)  # Evaluate on training data
#             test_acc = evaluate(data, data.test_mask)    # Evaluate on test data

#             losses.append(loss.item())
#             train_accurate.append(train_acc)
#             test_accurate.append(test_acc)
#             writer.writerow([epoch, loss.item(), train_acc, test_acc])

def train(data, patience=30):
    best_val_loss = float('inf')
    best_model = None
    patience_counter = 0
    
    losses, train_accurate, test_accurate = list(), list(), list()
    with open("GATv2_ACC.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Epoch', 'Loss', 'Train Accuracy', 'Test Accuracy'])

        for epoch in range(100):
            model.train()
            optimizer.zero_grad()
            out = model(data)
            loss = func.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            train_acc = evaluate(data, data.train_mask)  # Evaluate on training data
            val_loss = evaluate_loss(data, data.val_mask)  # Evaluate loss on validation data
            test_acc = evaluate(data, data.test_mask)    # Evaluate on test data

            # if val_loss < best_val_loss:
            #     best_val_loss = val_loss
            #     best_model = copy.deepcopy(model.state_dict())  # Save a copy of the current best model
            #     patience_counter = 0
            # else:
            #     patience_counter += 1
            
            # if patience_counter >= patience:
            #     print(f"Stopping early at epoch {epoch}")
            #     break

            losses.append(loss.item())
            train_accurate.append(train_acc)
            test_accurate.append(test_acc)
            writer.writerow([epoch, loss.item(), train_acc, test_acc])

    # model.load_state_dict(best_model)  # Load the best model after early stopping
    return losses, train_accurate, test_accurate

def evaluate_loss(data, mask):
    model.eval()
    with torch.no_grad():
        logits = model(data)
        loss = func.nll_loss(logits[mask], data.y[mask])
    return loss.item()

if __name__ == "__main__":
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Type of dataset[0]: ", type(dataset[0]))
    # dataset[0].x = normalize_features(dataset[0].x)
    features = dataset[0].x
    scaler = StandardScaler()
    scaled_features = torch.tensor(scaler.fit_transform(features), dtype=torch.float32)
    dataset[0].x = scaled_features
    
    model = GATv2_ten(dataset).to(device)
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    train(data)
