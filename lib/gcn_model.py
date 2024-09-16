from __future__ import print_function


__author__ = "Akash Vartak, Murad Khondoker"
__email__ = "akashvartak@umbc.edu, hossain10@umbc.edu"

# Standard library imports

# Standard library imports
from tqdm import tqdm
import os
import json

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Softmax

# PyG imports
from torch_geometric.nn import global_mean_pool, GraphConv

from lib.trafficnn import TrafficNN


def train_model(device, model, dataloader, optimizer, loss_function, epoch, logger):
    # set model in train mode
    model.train()

    enum_data = enumerate(dataloader)
    enum_data = tqdm(enum_data)

    total_loss = 0
    total_samples = 0
    correct = 0

    for batch_id, batch in enum_data:
        batch = batch.to(device)

        optimizer.zero_grad()

        # output = model(batch.x.float(), batch.edge_index, batch.batch)
        output = model(batch.x.float(), batch.edge_attr.float(), batch.edge_index, batch.batch)

        # output_cpu = output.detach().cpu()
        prediction = torch.argmax(output, dim=1)

        loss = torch.sqrt( loss_function(output, batch.y) )
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        batch_total = output.shape[0]
        total_samples += batch_total

        batch_correct = prediction.eq((batch.y).view_as(prediction)).sum().item()
        correct += batch_correct

        logger.info("[TRAIN] Train Epoch: {} | Batch: {}".format(epoch, batch_id + 1))
        logger.info("[TRAIN]   Accuracy: {}/{} ({:.2f}%)".format(correct, total_samples, 100. * (correct/total_samples)))
        logger.info("[TRAIN]   Average loss: {:.4f}".format(total_loss/(batch_id + 1)))
        logger.info("[TRAIN]   Batch loss: {:.4f}".format( loss.item() ))


        # if(math.isnan( total_loss/(batch_id + 1) )):
        #     print(f"output: {output}")
        #     print(f"batch.y: {batch.y}")
        #     print(f"loss_function: {loss_function(output, batch.y)}")
        #     print(f"loss.item(): {loss.item()}")
        #     print(f"total_loss: {total_loss}")
        #     print(f"batch_id+1: {batch_id+1}")
        #     sys.exit(0)

        # continue_input = input("continue [y/n] ? ")
        # if(continue_input!='y'):
        #     sys.exit(0)

    return None


def test_model(device, model, dataloader, optimizer, logger):
    # set model in train mode
    model.eval()

    enum_data = enumerate(dataloader)
    enum_data = tqdm(enum_data)

    prob_trojaned = 0.0

    with torch.no_grad():
        for batch_id, batch in enum_data:
            batch = batch.to(device)

            # output = model(batch.x.float(), batch.edge_index, batch.batch)
            output = model(batch.x.float(), batch.edge_attr.float(), batch.edge_index, batch.batch)

            output_cpu = output.detach().cpu()
            prediction = torch.argmax(output, dim=1)
            prob_clean = output_cpu[0][0].item()
            prob_trojaned = output_cpu[0][1].item()

            logger.info(f"[TEST] Output of model: {output_cpu}")
            logger.info("[TEST] Probability that model is Clean: {:.16f}".format( prob_clean ))
            logger.info("[TEST] Probability that model is Trojaned: {:.16f}".format( prob_trojaned ))
            logger.info(f"[TEST] Model Prediction: {prediction}")
    return prob_trojaned


def load_trafficnn_model(model_filepath: str) -> (dict, str):
    """Load a model given a specific model_path.

    Args:
        model_filepath: str - Path to model.pt file

    Returns:
        model - Torch model
    """
    conf_filepath = os.path.join(os.path.dirname(model_filepath), 'reduced-config.json')
    with open(conf_filepath, 'r') as f:
        full_conf = json.load(f)

    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if (cuda_available) else "cpu")

    model = TrafficNN(full_conf['img_resolution'] ** 2, full_conf)
    model.model.load_state_dict(torch.load(model_filepath, map_location = device))
    model.model.to(model.device).eval()
    # model = torch.load(model_filepath)

    return model.model


class GraphConvModel_WithEdgeFeat(nn.Module):
    def __init__(self, hidden_channels, num_features, num_classes):
        super(GraphConvModel_WithEdgeFeat, self).__init__()

        self.conv1 = GraphConv(num_features, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

        self.softmax = Softmax()

    def forward(self, x, edge_attr, edge_index, batch):
        x = self.conv1(x, edge_index, edge_attr)
        x = x.relu()

        x = self.conv2(x, edge_index, edge_attr)
        x = x.relu()

        x = self.conv3(x, edge_index, edge_attr)

        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        x = self.softmax(x)

        return x
