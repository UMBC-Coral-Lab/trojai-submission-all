import sys
import json
import logging
import os
import pickle
from os import listdir, makedirs
from os.path import join, exists, basename

import numpy as np
from tqdm import tqdm

from utils.abstract import AbstractDetector

from archs import Net2, Net3, Net4, Net5, Net6, Net7, Net2r, Net3r, Net4r, Net5r, Net6r, Net7r, Net2s, Net3s, Net4s, Net5s, Net6s, Net7s
import torch
import torch.optim as optim
from torch.optim import lr_scheduler

from lib.copy_all_models import copy_all_models
from lib.mlp_data import custom_collate, get_sampled_data, MLPDataset
from lib.mlp_model import MLPClassifier, test_model, train_model


class Detector(AbstractDetector):
    def __init__(self, metaparameter_filepath, learned_parameters_dirpath, scale_parameters_filepath):
        """Detector initialization function.

        Args:
            metaparameter_filepath: str - File path to the metaparameters file.
            learned_parameters_dirpath: str - Path to the learned parameters directory.
            scale_parameters_filepath: str - File path to the scale_parameters file.
        """
        self.logger = logging.getLogger('logger-mlp_pipeline')
        logging.basicConfig(level = logging.INFO,
                            stream = sys.stdout,
                            format = "%(asctime)s [%(levelname)s] %(message)s")
        self.logger.info("[DETECTOR] %s launched" % sys.argv[0])

        self.logger.info("[DETECTOR] Constructor 'init' called.")

        self.scale_parameters_filepath = scale_parameters_filepath
        self.metaparameter_filepath = metaparameter_filepath
        self.learned_parameters_dirpath = learned_parameters_dirpath
        self.model_filepath = join(self.learned_parameters_dirpath, "mlp_classifier_model.pt")

        metaparameters = json.load( open(self.metaparameter_filepath, "r") )

        self.train_params = {
            "epochs": metaparameters["train_epochs"],
            "batch_size": metaparameters["train_batch_size"],
            "lr" : 0.001,
            "gamma" : 0.7,
            "use_cuda" : True,
        }

        self.logger.info(f"[INIT] self.scale_parameters_filepath : {self.scale_parameters_filepath}")
        self.logger.info(f"[INIT] self.metaparameter_filepath : {self.metaparameter_filepath}")
        self.logger.info(f"[INIT] self.learned_parameters_dirpath : {self.learned_parameters_dirpath}")
        self.logger.info(f"[INIT] self.model_filepath : {self.model_filepath}")
        self.logger.info(f"[INIT] self.train_params : {self.train_params}")


    def write_metaparameters(self):
        self.logger.info("[WRITE PARAMS] Function 'write_metaparameters' called.")

        metaparameters = {
            "train_epochs": self.train_params["epochs"],
            "train_batch_size": self.train_params["batch_size"],
            "lr" : 0.001,
            "gamma" : 0.7,
            "use_cuda" : True,
        }

        with open(join( self.learned_parameters_dirpath, basename(self.metaparameter_filepath) ), "w") as fp:
            json.dump(metaparameters, fp)


    def automatic_configure(self, models_dirpath: str):
        """Configuration of the detector iterating on some of the parameters from the
        metaparameter file, performing a grid search type approach to optimize these
        parameters.

        Args:
            models_dirpath: str - Path to the list of model to use for training
        """
        self.logger.info("[AUTO CONFIG] Function 'automatic_configure' called.")

        for e in range(15, 31, 2):
            for b in range(1, 65, 8):
                self.train_params["epochs"] = e
                self.train_params["batch_size"] = b

                self.logger.info(f"[AUTO CONFIG] New Epoch: {e}, new batch size: {b}")

                self.manual_configure(models_dirpath)


    def manual_configure(self, models_dirpath: str):
        """Configuration of the detector using the parameters from the metaparameters
        JSON file.

        Args:
            models_dirpath: str - Path to the list of model to use for training
        """
        self.logger.info("[MANUAL CONFIG] Function 'manual_configure' called.")

        self.scratch_dirpath = "scratch/"

        # Create the learned parameter folder if needed
        if not exists(self.learned_parameters_dirpath):
            makedirs(self.learned_parameters_dirpath)

        if not exists(self.scratch_dirpath):
            makedirs(self.scratch_dirpath)

        cuda_available = torch.cuda.is_available()
        device = torch.device("cuda" if (self.train_params["use_cuda"] and cuda_available) else "cpu")

        # get all models in models_dirpath and loop over them to generate the dataset
        # dest_models_path will be 'self.scratch_dirpath/train/'
        # csv_file will be 'all_models.csv'
        dest_models_path, csv_file = copy_all_models(models_dirpath, self.scratch_dirpath, self.logger, type = "train")

        # create and load dataset
        mlp_dataset = MLPDataset(device, dest_models_path, csv_file, self.logger)

        # create train sample
        tr_sampler, te_sampler = get_sampled_data(len(mlp_dataset), 1, self.logger)

        # create train DataLoader
        train_loader = torch.utils.data.DataLoader( dataset = mlp_dataset, batch_size = self.train_params["batch_size"], sampler = tr_sampler, collate_fn = custom_collate)

        # create MLP model
        model = MLPClassifier().to(device)

        self.logger.info(f"[MANUAL CONFIG] Meta-classifier Model: {model.__class__.__name__}")
        self.logger.info(f"[MANUAL CONFIG] Architecture: \n{model}")

        # create optimizer parameter
        optimizer = optim.Adam(model.parameters(), lr = self.train_params["lr"])

        # create learning rate scheduler
        scheduler = lr_scheduler.StepLR(optimizer, step_size = 1, gamma = self.train_params["gamma"])

        for epoch in range(1, self.train_params["epochs"]+1):
            # train model with train set
            train_model(device, model, train_loader, optimizer, epoch, self.logger)

            # learning rate scheduler step
            scheduler.step()

            print("")


        # save model in learned_parameters_dirpath
        save_parameters = { "model_state_dict": model.state_dict()
                            , "optimizer_state_dict": optimizer.state_dict()
                            , "epoch": self.train_params["epochs"]
                            , "batch_size": self.train_params["batch_size"] }

        self.logger.info(f"[MANUAL CONFIG] Trained MLP meta-classifier model saved at path '{self.model_filepath}'.")
        torch.save(save_parameters, self.model_filepath )

        self.write_metaparameters()
        self.logger.info("[MANUAL CONFIG] Configuration done!")


    def infer(self, model_filepath, result_filepath, scratch_dirpath, examples_dirpath, round_training_dataset_dirpath):
        """Method to predict wether a model is poisoned (1) or clean (0).

        Args:
            model_filepath:
            result_filepath:
            scratch_dirpath:
            examples_dirpath:
            round_training_dataset_dirpath:
        """
        self.logger.info("[INFER] Function 'infer' called.")

        cuda_available = torch.cuda.is_available()
        device = torch.device("cuda" if (self.train_params["use_cuda"] and cuda_available) else "cpu")

        # get all models in models_dirpath and loop over them to generate the dataset
        # dest_models_path will be 'self.scratch_dirpath/test/'
        # csv_file will be 'all_models.csv'
        # dest_models_path, csv_file = copy_all_models(model_filepath, scratch_dirpath, self.logger, type = "test")

        # create and load dataset
        mlp_dataset = MLPDataset(device, model_filepath, "", self.logger, type = "test")

        # create train sample
        tr_sampler, te_sampler = get_sampled_data(len(mlp_dataset), 1, self.logger)

        # create train DataLoader
        test_loader = torch.utils.data.DataLoader( dataset = mlp_dataset, batch_size = self.train_params["batch_size"], sampler = tr_sampler, collate_fn = custom_collate)

        # create MLP model
        model = MLPClassifier().to(device)

        # load previously saved MLP model
        self.logger.info(f"[INFER] Loading meta-classifier model from path: {self.model_filepath}")

        checkpoint = torch.load( self.model_filepath )
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)

        self.logger.info(f"[INFER] Loaded meta-classifier model: {model.__class__.__name__}")
        self.logger.info(f"[INFER] Architecture: \n{model}")

        # create optimizer parameter
        optimizer = optim.Adam(model.parameters(), lr = self.train_params["lr"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # create learning rate scheduler
        scheduler = lr_scheduler.StepLR(optimizer, step_size = 1, gamma = self.train_params["gamma"])

        # test on the test set for a single epoch
        probability_trojaned = test_model(device, model, test_loader, self.logger, epoch = None)

        self.logger.info(f"[INFER] Writing output probability to file: '{result_filepath}'")

        probability_trojaned_str = str(probability_trojaned)
        with open(result_filepath, "w") as fp:
            fp.write(probability_trojaned_str)

        self.logger.info("[INFER] Trojan probability: %s", probability_trojaned_str)
