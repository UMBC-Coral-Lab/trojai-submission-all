from __future__ import print_function


__author__ = "Akash Vartak"
__email__ = "akashvartak@umbc.edu"


# Standard library imports
import os
import shutil
from typing import Tuple

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

# PyG imports
from torch_geometric.loader import DataLoader

from lib.copy_all_models import copy_all_models
from lib.gcn_dataset import GraphsDataset_Configure, GraphsDataset_Inference
from lib.gcn_model import GraphConvModel_WithEdgeFeat, test_model, train_model


"""
Function to return a dictionary of constants
"""
def get_constants():
    # Constants declaration:
    CONSTANTS = {
            "use_cuda" : True,
            "dataset_name" : 12,
            "hidden_channels" : 64,
            "num_classes" :  2,
            "lr" : 0.001,
            "gamma" : 0.7
        }

    return CONSTANTS



# Print some diagnostic dataset information
def print_dataset_stats(dataset, idx, logger):
    logger.info(f"[SAMPLE]  Dataset: {dataset}:")
    logger.info("[SAMPLE] =============================================================")
    logger.info(f"[SAMPLE] Number of graphs: {len(dataset)}")
    logger.info(f"[SAMPLE] Number of features: {dataset.num_features}")
    logger.info(f"[SAMPLE] Number of classes: {get_constants()['num_classes']}")

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



"""
Get the dataset name as a string
Args:
    dataset_identifier:
        The number identifying which node features vector to use
"""
def get_dataset_name_string(dataset_identifier: int) -> Tuple[str, str]:
    dataset_name_short = ""
    dataset_name_string = ""

    if(dataset_identifier == 1):
        dataset_name_short = "1x3"
        dataset_name_string = "1-LR-ND"
    elif(dataset_identifier == 2):
        dataset_name_short = "1x5_2"
        dataset_name_string = "mean-min-max-sum-bias"
    elif(dataset_identifier == 3):
        dataset_name_short = "1x5"
        dataset_name_string = "1-LR-mean-min-max"
    elif(dataset_identifier == 4):
        dataset_name_short = "1x7"
        dataset_name_string = "1-LR-mean-min-max-histbin-histboundary"
    elif(dataset_identifier == 5):
        dataset_name_short = "1x7_2"
        dataset_name_string = "1-LR-mean-min-max-ND-sum"
    elif(dataset_identifier == 6):
        dataset_name_short = "1x9"
        dataset_name_string = "1-LR-mean-min-max-histbin-histboundary-ND-sum"
    elif(dataset_identifier == 7):
        dataset_name_short = "1x7_3"
        dataset_name_string = "1-mean-min-max-histbin-histboundary-sum"
    elif(dataset_identifier == 8):
        dataset_name_short = "1x8"
        dataset_name_string = "mean-min-max-25histbin-25histboundary-ND-sum"
    elif(dataset_identifier == 9):
        dataset_name_short = "1x8_2"
        dataset_name_string = "LR-mean-min-max-histbin-histboundary-sum-bias"
    elif(dataset_identifier == 10):
        dataset_name_short = "1x7_4"
        dataset_name_string = "mean-min-max-histbin-histboundary-sum-bias"
    elif(dataset_identifier == 11):
        dataset_name_short = "1x4"
        dataset_name_string = "nodecount-histbin-histboundary-ND"
    elif(dataset_identifier == 12):
        dataset_name_short = "1x9_2"
        dataset_name_string = "nodecount-mean-min-max-histbin-histboundary-ND-sum-bias"

    return dataset_name_short, dataset_name_string



"""
Function that will be called for inference mode
Args:
    model_filepath (str):
        The *.pt model to be evaluated.
    result_filepath (str):
        The *.txt file to which the probability of trojaned of model 'model_filepath' is to be written.
    learned_parameters_dirpath (str):
        Directory path of the GCN model to use for classification.
    scratch_dirpath (str):
        Directory path of scratch space for temporary usage. Assumption is that the scratch space is wiped clean and is empty.
    epochs (int):
        Number of epochs to be run.
    batch_size (int):
        Batch size for dataloader
    step_size (int):
        Step size for LR decay.
    logger:
        Logger object to log events.
"""
def inference_mode(
        model_filepath: str,
        result_filepath: str,
        learned_parameters_dirpath: str,
        scratch_dirpath: str,
        epochs: int, batch_size: int, step_size: int,
        logger
    ):
    # get all constants
    _CONSTANTS = get_constants()

    # CPU/GPU device to use
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if (_CONSTANTS["use_cuda"] and cuda_available) else "cpu")

    # get node features vector to use
    dataset_name_short, dataset_name_string = get_dataset_name_string( _CONSTANTS["dataset_name"] )

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
                                        logger = logger )

    test_loader = DataLoader(dataset, batch_size = batch_size, shuffle = True)

    # load model from learned_parameters_dirpath
    load_model_path = os.path.join(learned_parameters_dirpath, "gcn_classifier_model.pt")
    logger.info(f"[INFERENCE] Loading meta-classifier GCN model: '{load_model_path}'")

    model = GraphConvModel_WithEdgeFeat(hidden_channels = _CONSTANTS["hidden_channels"], num_features = dataset.num_features, num_classes = _CONSTANTS["num_classes"])

    checkpoint = torch.load( load_model_path )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    # create optimizer parameter
    optimizer = optim.Adam(model.parameters(), lr = _CONSTANTS["lr"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # call test function with gcn classifier and converted graph
    prob_trojaned = test_model(device, model, test_loader, optimizer, logger)

    logger.info(f"[INFERENCE] Writing output probability to file '{result_filepath}'")
    # write the probability to result_filepath
    with open( result_filepath, "w" ) as output_file:
        output_file.write("{:.16f}".format( prob_trojaned ))



"""
Function that will be called for configure  mode
Args:
    model_filepath (str):
        The *.pt model to be evaluated.
    result_filepath (str):
        The *.txt file to which the probability of trojaned of model 'model_filepath' is to be written.
    learned_parameters_dirpath (str):
        Directory path of the GCN model to use for classification.
    scratch_dirpath (str):
        Directory path of scratch space for temporary usage. Assumption is that the scratch space is wiped clean and is empty.
    epochs (int):
        Number of epochs to be run.
    batch_size (int):
        Batch size for dataloader
    step_size (int):
        Step size for LR decay.
    logger:
        Logger object to log events.
"""
def configure_mode(
        configure_models_dirpath: str,
        learned_parameters_dirpath: str,
        scratch_dirpath: str,
        epochs: int, batch_size: int, step_size: int,
        logger
    ):
    # get all constants
    _CONSTANTS = get_constants()

    # CPU/GPU device to use
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if (_CONSTANTS["use_cuda"] and cuda_available) else "cpu")

    # get node features vector to use
    dataset_name_short, dataset_name_string = get_dataset_name_string( _CONSTANTS["dataset_name"] )

    # delete any previous inference datasets:
    graph_dataset_root = os.path.join(scratch_dirpath, "configure_graph_dataset", dataset_name_short)

    if os.path.exists(graph_dataset_root) and os.path.isdir(graph_dataset_root):
        shutil.rmtree(graph_dataset_root)

    # load model_filepath *.pt file, convert it to graph and store in scratch space
    processed_path = os.path.join(graph_dataset_root, "processed/")
    raw_path = os.path.join(graph_dataset_root, "raw/")

    if not os.path.exists( processed_path ):
        os.makedirs( processed_path )
    if not os.path.exists( raw_path ):
        os.makedirs( raw_path )

    # get all models in configure_models_dirpath and loop over them to generate the dataset
    dest_models_path, csv_file = copy_all_models(configure_models_dirpath, scratch_dirpath, logger)

    dataset = GraphsDataset_Configure( scratch_dirpath,
                                        device,
                                        root = graph_dataset_root,
                                        csv_file = csv_file,
                                        dataset_name_short = dataset_name_short,
                                        dataset_name_string = dataset_name_string,
                                        logger = logger )
    print_dataset_stats(dataset, 0, logger)

    # create dataloader
    train_loader = DataLoader(dataset, batch_size = batch_size, shuffle = True)

    # create gcn model
    model = GraphConvModel_WithEdgeFeat(hidden_channels = _CONSTANTS["hidden_channels"], num_features = dataset.num_features, num_classes = _CONSTANTS["num_classes"])
    model = model.to(device)

    # create loss function object
    loss_function = nn.CrossEntropyLoss()

    # create optimizer parameter
    optimizer = optim.Adam(model.parameters(), lr = _CONSTANTS["lr"])

    # create learning rate scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size = step_size, gamma = _CONSTANTS["gamma"])

    # call train function with gcn classifier and converted graph
    for epoch in range(1, epochs+1):
        # train model with train set
        train_model(device, model, train_loader, optimizer, loss_function, epoch, logger)

        # learning rate scheduler step
        scheduler.step()

        print("")

    # save model in learned_parameters_dirpath
    save_parameters = { "model_state_dict": model.state_dict()
                        , "optimizer_state_dict": optimizer.state_dict()
                        , "epoch": epochs
                        , "batch_size": batch_size
                        , "step_size": step_size }
    torch.save(save_parameters, os.path.join(learned_parameters_dirpath, "gcn_classifier_model.pt") )
