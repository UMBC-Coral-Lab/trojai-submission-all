from __future__ import print_function


__author__ = "Akash Vartak"
__email__ = "akashvartak@umbc.edu"

# Standard library imports
import sys
from tqdm import tqdm

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F


# function to train model
def train_model(device, model, dataloader, optimizer, epoch, logger):
    # set model in train mode
    model.train()

    enum_data = enumerate(dataloader)
    enum_data = tqdm(enum_data)

    total_loss = 0
    total_samples = 0
    correct = 0

    for batch_id, (model_weights, target) in enum_data:

        model_weights, target = model_weights.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(model_weights)
        output_cpu = output.detach().cpu()
        prediction = torch.argmax(output, dim=1)

        # print(f"Targets: {target}")
        # print(f"Output: {output_cpu}")
        # print(f"Prediction: {prediction}")

        loss = F.cross_entropy(output, target = target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_samples += output.shape[0]

        correct += prediction.eq(target.view_as(prediction)).sum().item()

        logger.info("[TRAIN] Train Epoch: {} Batch: {} | [{}/{}]".format(epoch, batch_id + 1, batch_id * model_weights.shape[0], total_samples))
        logger.info("[TRAIN]   Accuracy: {}/{} ({:.2f}%)".format(correct, total_samples, 100. * correct/total_samples))
        logger.info("[TRAIN]   Average loss: {:.4f}".format(total_loss/(batch_id + 1)))

        # continue_input = input("continue [y/n] ? ")
        # if(continue_input!='y'):
        #     sys.exit(0)

    return None


# function to test model
def test_model(device, model, dataloader, logger, epoch = None):
    # set model in eval mode for test
    model.eval()

    enum_data = enumerate(dataloader)
    enum_data = tqdm(enum_data)

    prob_trojaned = 0.0

    with torch.no_grad():
        for batch_id, (model_weights, target) in enum_data:
            model_weights, target = model_weights.to(device), target.to(device)

            output = model(model_weights)
            output_cpu = output.detach().cpu()
            prediction = torch.argmax(output, dim=1)

            prob_clean = output_cpu[0][0].item()
            prob_trojaned = output_cpu[0][1].item()

            logger.info(f"[TEST] Output of model: {output_cpu}")
            logger.info("[TEST] Probability that model is Clean: {:.16f}".format( prob_clean ))
            logger.info("[TEST] Probability that model is Trojaned: {:.16f}".format( prob_trojaned ))
            logger.info(f"[TEST] Model Prediction: {prediction}")

    return prob_trojaned


class MLPClassifier(nn.Module):
    def __init__(self):
        super(MLPClassifier, self).__init__()
        self.relu = nn.ReLU(inplace = True)

        self.fc1 = nn.Linear(in_features = 200, out_features = 1024)
        self.fc2 = nn.Linear(in_features = 1024, out_features = 512)
        self.fc3 = nn.Linear(in_features = 512, out_features = 2)

        self.softmax = nn.Softmax()

    def forward(self, input):
        fc_out = self.fc1(input)
        relu_out = self.relu(fc_out)
        fc_out = self.fc2(relu_out)
        relu_out = self.relu(fc_out)
        fc_out = self.fc3(relu_out)
        softmax_out = self.softmax(fc_out)

        return softmax_out
