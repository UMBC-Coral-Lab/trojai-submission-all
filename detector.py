# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

# Standard library imports
import json
import jsonpickle
import logging
import os
from os import makedirs
from os.path import basename, exists, join
import pickle
import sys
import shutil

import numpy as np
import pandas as pd

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

# RL Imports
import torch_ac
import gym
from gym_minigrid.wrappers import ImgObsWrapper

# PyG imports
from torch_geometric.loader import DataLoader

# Local Imports
from utils.abstract import AbstractDetector
from utils.models import load_model, load_models_dirpath

from lib.copy_all_models import copy_all_models
from lib.gcn_dataset import GraphsDataset_Configure, GraphsDataset_Inference
from lib.gcn_model import GraphConvModel_WithEdgeFeat, test_model, train_model


class Detector(AbstractDetector):
    def __init__(self, metaparameter_filepath, learned_parameters_dirpath):
        """Detector initialization function.

        Args:
            metaparameter_filepath: str - File path to the metaparameters file.
            learned_parameters_dirpath: str - Path to the learned parameters directory.
            scale_parameters_filepath: str - File path to the scale_parameters file.
        """
        self.logger = logging.getLogger('logger-svm_pipeline')
        logging.basicConfig(level = logging.INFO,
                            stream = sys.stdout,
                            format = "%(asctime)s [%(levelname)s] %(message)s")

        self.logger.info("[DETECTOR] %s launched" % sys.argv[0])
        self.logger.info("[DETECTOR] Constructor 'init' called.")

        self.metaparameter_filepath = metaparameter_filepath
        self.learned_parameters_dirpath = learned_parameters_dirpath
        self.model_filepath = join(self.learned_parameters_dirpath, "gcn_classifier_model.pt")

        metaparameters = json.load( open(self.metaparameter_filepath, "r") )

        self.train_params = {
            "train_epochs" : metaparameters["train_epochs"], 
            "train_batch_size" : metaparameters["train_batch_size"], 
            "train_feature_vector" : metaparameters["train_feature_vector"], 
            "use_cuda" : True, 
            "hidden_channels" : 64, 
            "num_classes" :  2, 
            "lr" : 0.001, 
            "gamma" : 0.7, 
            "step_size" : 2
        }

        self.logger.info(f"[INIT] self.metaparameter_filepath : {self.metaparameter_filepath}")
        self.logger.info(f"[INIT] self.learned_parameters_dirpath : {self.learned_parameters_dirpath}")
        self.logger.info(f"[INIT] self.model_filepath : {self.model_filepath}")
        self.logger.info(f"[INIT] self.train_params : {self.train_params}")


    # def get_constants():
    #     """
    #     Function to return a dictionary of constants
    #     """

    #     CONSTANTS = {
    #             "use_cuda" : True,
    #             "hidden_channels" : 64,
    #             "num_classes" :  2,
    #             "lr" : 0.001,
    #             "gamma" : 0.7,
    #             "use_cuda" : True
    #         }

    #     return CONSTANTS


    # Get the dataset name as a string:
    def get_dataset_name_string(self, dataset_identifier: int):
        dataset_name_short = ""
        dataset_name_string = ""

        dataset_dict = {
            4 : ["1x7",     "1-LR-mean-min-max-histbin-histboundary"],
            5 : ["1x7_2",   "1-LR-mean-min-max-ND-sum"],
            6 : ["1x9",     "1-LR-mean-min-max-histbin-histboundary-ND-sum"],
            7 : ["1x7_3",   "1-mean-min-max-histbin-histboundary-sum"],
            8 : ["1x8",     "mean-min-max-25histbin-25histboundary-ND-sum"],
            9 : ["1x8_2",   "LR-mean-min-max-histbin-histboundary-sum-bias"],
            10 : ["1x7_4",  "mean-min-max-histbin-histboundary-sum-bias"],
            11 : ["1x4",    "nodecount-histbin-histboundary-ND"],
            12 : ["1x9_2",  "nodecount-mean-min-max-histbin-histboundary-ND-sum-bias"]

        }

        for k, v in dataset_dict.items():
            if(k == dataset_identifier):
                dataset_name_short = v[0]
                dataset_name_string = v[1]

        return dataset_name_short, dataset_name_string


    # Print some diagnostic dataset information
    def print_dataset_stats(self, dataset, idx, logger):
        logger.info(f"[SAMPLE]  Dataset: {dataset}:")
        logger.info("[SAMPLE] =============================================================")
        logger.info(f"[SAMPLE] Number of graphs: {len(dataset)}")
        logger.info(f"[SAMPLE] Number of features: {dataset.num_features}")
        logger.info(f"[SAMPLE] Number of classes: {self.train_params['num_classes']}")

        # Gather some statistics about the first graph.
        data = dataset[idx]
        logger.info("[SAMPLE] =============================================================")
        logger.info(f"[SAMPLE] Example data object: {data}")
        logger.info(f"[SAMPLE] Number of nodes: {data.num_nodes}")
        logger.info(f"[SAMPLE] Number of edges: {data.num_edges}")
        logger.info(f"[SAMPLE] Average node degree: {data.num_edges / data.num_nodes:.2f}")
        logger.info(f"[SAMPLE] Has isolated nodes: {data.has_isolated_nodes()}")
        logger.info(f"[SAMPLE] Has self-loops: {data.has_self_loops()}")
        logger.info(f"[SAMPLE] Is undirected: {data.is_undirected()}")


    def write_metaparameters(self):
        self.logger.info("[WRITE PARAMS] Function 'write_metaparameters' called.")

        metaparameters = {
            "train_epochs": self.train_params["train_epochs"], 
            "train_batch_size": self.train_params["train_batch_size"], 
            "train_feature_vector": self.train_params["train_feature_vector"], 
            "use_cuda" : True, 
            "hidden_channels" : 64, 
            "num_classes" : 2, 
            "lr" : 0.001, 
            "gamma" : 0.7, 
            "step_size" : 2
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

        # max_test_accuracy = 0.0
        # for feat in range(25, 200, 5):
        #     self.logger.info(f"[AUTO CONFIG] New max features: {feat}")

        #     self.train_params["max_feats"] = feat
        #     max_test_accuracy = self.automatic_configure_best_test(models_dirpath, max_test_accuracy, test_size = 0.2) # self.manual_configure(models_dirpath)


        for e in range(20, 41, 2):
            for b in range(18, 37, 6):
                for f in range(4, 13, 1):
                    self.train_params["train_epochs"] = e
                    self.train_params["train_batch_size"] = b
                    self.train_params["train_feature_vector"] = f

                    self.logger.info(f"[AUTO CONFIG] New training parameters:")
                    self.logger.info(f"[AUTO CONFIG]    epoch: {self.train_params['train_epochs']}")
                    self.logger.info(f"[AUTO CONFIG]    batch size: {self.train_params['train_batch_size']}")
                    self.logger.info(f"[AUTO CONFIG]    node feature vector: {self.train_params['train_feature_vector']}")

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

        # get all constants
        # _CONSTANTS = get_constants()

        # CPU/GPU device to use
        cuda_available = torch.cuda.is_available()
        device = torch.device("cuda" if (self.train_params["use_cuda"] and cuda_available) else "cpu")

        # get node features vector to use
        dataset_name_short, dataset_name_string = self.get_dataset_name_string( int(self.train_params["train_feature_vector"]) )

        # delete any previous inference datasets:
        graph_dataset_root = os.path.join(self.scratch_dirpath, "configure_graph_dataset", dataset_name_short)

        if os.path.exists(graph_dataset_root) and os.path.isdir(graph_dataset_root):
            shutil.rmtree(graph_dataset_root)

        # load model_filepath *.pt file, convert it to graph and store in scratch space
        processed_path = os.path.join(graph_dataset_root, "processed/")
        raw_path = os.path.join(graph_dataset_root, "raw/")

        if not os.path.exists( processed_path ):
            os.makedirs( processed_path )
        if not os.path.exists( raw_path ):
            os.makedirs( raw_path )

        # get all models in models_dirpath and loop over them to generate the dataset
        dest_models_path, csv_file = copy_all_models(models_dirpath, self.scratch_dirpath, self.logger)

        dataset = GraphsDataset_Configure( self.scratch_dirpath,
                                            device,
                                            root = graph_dataset_root,
                                            csv_file = csv_file,
                                            dataset_name_short = dataset_name_short,
                                            dataset_name_string = dataset_name_string,
                                            logger = self.logger )
        self.print_dataset_stats(dataset, 0, self.logger)

        # create dataloader
        train_loader = DataLoader(dataset, batch_size = self.train_params["train_batch_size"], shuffle = True)

        # create gcn model
        model = GraphConvModel_WithEdgeFeat(hidden_channels = self.train_params["hidden_channels"], num_features = dataset.num_features, num_classes = self.train_params["num_classes"])
        model = model.to(device)

        # create loss function object
        loss_function = nn.CrossEntropyLoss()

        # create optimizer parameter
        optimizer = optim.Adam(model.parameters(), lr = self.train_params["lr"])

        # create learning rate scheduler
        scheduler = lr_scheduler.StepLR(optimizer, step_size = self.train_params["step_size"], gamma = self.train_params["gamma"])

        # call train function with gcn classifier and converted graph
        for epoch in range(1, self.train_params["train_epochs"]+1):
            # train model with train set
            train_model(device, model, train_loader, optimizer, loss_function, epoch, self.logger)

            # learning rate scheduler step
            scheduler.step()

            print("")

        # save model in self.model_filepath
        save_parameters = { "model_state_dict": model.state_dict()
                            , "optimizer_state_dict": optimizer.state_dict()
                            , "epoch": self.train_params["train_epochs"]
                            , "batch_size": self.train_params["train_batch_size"]
                            , "step_size": self.train_params["step_size"]
                            , "feature_vector": self.train_params["train_feature_vector"] }
        torch.save(save_parameters, self.model_filepath )
        self.logger.info(f"[MANUAL CONFIG] Trained GCN meta-classifier model saved at path '{self.model_filepath}'.")

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

        # get node features vector to use
        dataset_name_short, dataset_name_string = self.get_dataset_name_string( int(self.train_params["train_feature_vector"]) )

        # delete any previous inference datasets:
        graph_dataset_root = os.path.join(scratch_dirpath, "inference_graph_dataset", dataset_name_short)

        if os.path.exists(graph_dataset_root) and os.path.isdir(graph_dataset_root):
            shutil.rmtree(graph_dataset_root)

        # load model_filepath *.pt file, convert it to graph and store in scratch space
        processed_path = os.path.join(graph_dataset_root, "processed/")
        raw_path = os.path.join(graph_dataset_root, "raw/")

        if not os.path.exists( processed_path ):
            os.makedirs( processed_path )
        if not os.path.exists( raw_path ):
            os.makedirs( raw_path )

        dataset = GraphsDataset_Inference( model_filepath,
                                            device,
                                            root = graph_dataset_root,
                                            dataset_name_short = dataset_name_short,
                                            dataset_name_string = dataset_name_string,
                                            logger = self.logger )

        test_loader = DataLoader(dataset, batch_size = self.train_params["train_batch_size"], shuffle = True)

        # load model from self.model_filepath
        load_model_path = self.model_filepath
        self.logger.info(f"[INFERENCE] Loading meta-classifier GCN model: '{load_model_path}'")

        model = GraphConvModel_WithEdgeFeat(hidden_channels = self.train_params["hidden_channels"], num_features = dataset.num_features, num_classes = self.train_params["num_classes"])

        checkpoint = torch.load( load_model_path )
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)

        # create optimizer parameter
        optimizer = optim.Adam(model.parameters(), lr = self.train_params["lr"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # call test function with gcn classifier and converted graph
        prob_trojaned = test_model(device, model, test_loader, optimizer, self.logger)

        self.logger.info(f"[INFERENCE] Writing output probability to file '{result_filepath}'")
        # write the probability to result_filepath
        with open( result_filepath, "w" ) as output_file:
            output_file.write("{:.16f}".format( prob_trojaned ))

        self.logger.info("[INFER] Trojan probability: %s", prob_trojaned)


    # def automatic_configure_best_test(self, models_dirpath: str, max_prev_test_accuracy: float, test_size: float):
    #     """
    #     Args:
    #         models_dirpath: str - Path to the list of model to use for training
    #     """
    #     self.logger.info(f"[BEST CONFIG] Function 'automatic_configure_best_test' called with test_size: {test_size}.")

    #     self.scratch_dirpath = "scratch/"

    #     # Create the learned parameter folder if needed
    #     if not exists(self.learned_parameters_dirpath):
    #         makedirs(self.learned_parameters_dirpath)

    #     if not exists(self.scratch_dirpath):
    #         makedirs(self.scratch_dirpath)

    #     cuda_available = torch.cuda.is_available()
    #     device = torch.device("cuda" if (self.train_params["use_cuda"] and cuda_available) else "cpu")

    #     # load csv as dataframe
    #     df = pd.read_csv(os.path.join(models_dirpath, 'METADATA.csv'))

    #     models = load_models(models_dirpath, df)
    #     X = [featurize_model(model) for model in  models]
    #     y = df['ground_truth'].values

    #     # Train on all of the data
    #     clf = SVC(
    #         C = self.train_params["C"],
    #         kernel = self.train_params["kernel"],
    #         gamma = self.train_params["gamma"],
    #         probability = True
    #     )
    #     clf, sorted_idx, train_accuracy, test_accuracy = fit(clf, X, y, self.logger, test_size = test_size, max_feats = self.train_params["max_feats"])

    #     if(test_accuracy > max_prev_test_accuracy):
    #         max_prev_test_accuracy = test_accuracy
    #         self.write_metaparameters()

    #         # save model to self.model_filepath and sorted_idx to self.sorted_idx_filepath
    #         self.logger.info(f"[BEST CONFIG] Trained SVM meta-classifier model saved at path '{self.model_filepath}'.")
    #         pickle.dump(clf, open(self.model_filepath, "wb"))

    #         self.logger.info(f"[BEST CONFIG] Selected features for meta-classifier model saved at path '{self.sorted_idx_filepath}'.")
    #         np.save(self.sorted_idx_filepath, sorted_idx)

    #     self.logger.info("[BEST CONFIG] Configuration done!")

    #     return max_prev_test_accuracy
