from __future__ import print_function


__author__ = "Akash Vartak"
__email__ = "akashvartak@umbc.edu"

# Standard library imports
import argparse
import csv
import os

import numpy as np

# PyTorch imports
import torch
from torch.utils.data import Dataset, SubsetRandomSampler


"""
Custom collate function to manage batches
NOTE: The order of items is: data, target, poisoned or not, etc.
"""
# custom collate function to manage batches
def custom_collate(batch):
    model_weights = torch.FloatTensor([item[0] for item in batch])
    model_poisoned = torch.LongTensor([item[1] for item in batch])
    return [model_weights, model_poisoned]


"""
Generate random subset samplers given the size of the data and the split.
The setup assumes there is a single dataset and a subsection is split into
train and test.
NOTE: The ratios must sum to 1.
"""
def get_sampled_data(data_size, train_split, logger):

    num_tr = int(data_size * train_split)
    num_te = int(data_size * (1-train_split))

    # Indices.
    arr = np.array(range(data_size))
    np.random.shuffle(arr) # Shuffle the array.
    tr_id = arr[0: num_tr]
    te_id = arr[num_tr: ]

    logger.info("Train samples from 0 to {}".format(num_tr))
    logger.info("Test samples from {} to {}".format(num_tr, data_size))

    return SubsetRandomSampler(tr_id), SubsetRandomSampler(te_id)


"""
Load the dataset required for sanity check.
The dataset provides samples as tuples of:
- model weights
- whether cnn model is trojaned (value: yes/no)
"""
class MLPDataset(Dataset):
    def __init__(self, device, data_path, csv_name, logger, type = "train"):
        self.model_weights = []
        self.model_poisoned = []

        if(type == "train"):
            with open(data_path + csv_name, "r") as datacsv:
                csv_reader = csv.DictReader(datacsv)
                for row in csv_reader:
                    row_cnn_model_name = row["model_name"]
                    row_model_poisoned = 1 if(row["poisoned"].lower() == "yes") else 0

                    logger.info(f"[MLPDataset] Loading model '{os.path.join(data_path, row_cnn_model_name)}'")

                    model = torch.load( os.path.join(data_path, row_cnn_model_name), map_location = device)
                    model_params = list( model.parameters() )
                    model_params_len = len(model_params)

                    # get concatenated fc weights from the last 1 or 2 fc layers. The weights are at index length-2 and length-4:
                    # model_param_fc1 = model_params[4].detach().numpy()
                    model_param_fc2 = model_params[model_params_len-2].cpu().detach().numpy()

                    model_param_fc = np.concatenate((model_param_fc2), axis=None)

                    self.model_weights.append( model_param_fc )
                    self.model_poisoned.append( row_model_poisoned )
        elif(type == "test"):
            model = torch.load( data_path, map_location = device)
            model_params = list( model.parameters() )
            model_params_len = len(model_params)

            # get concatenated fc weights from the last 1 or 2 fc layers. The weights are at index length-2 and length-4:
            # model_param_fc1 = model_params[4].detach().numpy()
            model_param_fc2 = model_params[model_params_len-2].cpu().detach().numpy()

            model_param_fc = np.concatenate((model_param_fc2), axis=None)

            self.model_weights.append( model_param_fc )

            # append anything as the label since for test models, the label is not known:
            self.model_poisoned.append( 0 )



    # override required methods:
    # get length of dataset:
    def __len__(self):
        return len(self.model_weights)

    # get ith item from dataset
    def __getitem__(self, idx):
        idx_model_weights = self.model_weights[idx]
        idx_model_poisoned = self.model_poisoned[idx]

        return idx_model_weights, idx_model_poisoned
