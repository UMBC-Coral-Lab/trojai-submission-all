from __future__ import print_function


__author__ = "Akash Vartak"
__email__ = "akashvartak@umbc.edu"

# Standard library imports
import os
import shutil
from tqdm import tqdm
from typing import Callable, Optional

import numpy as np
import pandas as pd

# PyTorch imports
import torch

# PyG imports
import networkx as nx
from torch_geometric.data import Data, Dataset


"""
Load the dataset required for GCN.
This dataset provides graphical representation of the CNN with node features AND edge features.
"""
class GraphsDataset_Inference(Dataset):
    def __init__(self,
        model_filepath: str,
        device,
        root: Optional[str] = None,
        dataset_name_short: Optional[str] = None,
        dataset_name_string: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        logger = None
    ):
        # declare the necessary paths
        self._root = root
        self._model_path = model_filepath
        self._dataset_name_short = dataset_name_short
        self._dataset_name_string = dataset_name_string
        self._model_device = device
        self._print_example_node_feature = False
        self._logger = logger

        super(GraphsDataset_Inference, self).__init__(root, transform, pre_transform, pre_filter)



    @property
    def raw_file_names(self):
        return self._model_path


    @property
    def processed_file_names(self):
        processed_files = [f"data_{self._dataset_name_short}_0.pt"]
        return processed_files


    def download(self):
        shutil.copy2( self._model_path, os.path.join(self._root, "raw/") )


    def process(self):
        self._logger.info(f"[DATASET] ---- Converting {self._model_path} ----")


        self._logger.info("[DATASET] Getting model weights")
        # get cnn weights from npy file
        model_weights, model_fc_left_bias, model_fc_right_bias = self._get_model_weights_as_tensor( self._model_path )


        self._logger.info("[DATASET] Generating dummy node ids")
        # create psuedo node ids to create graph
        dummy_fc_fc1, dummy_fc_fc2 = self._get_dummy_node_ids( model_weights )


        self._logger.info("[DATASET] Generating adjacency matrix")
        # get adjacency matrix
        adjacency_list, adjacency_nparray = self._get_adjacency_matrix( len(dummy_fc_fc1), len(dummy_fc_fc2), model_weights )


        self._logger.info("[DATASET] Generating graph G")
        # create graph from nodes weights

        G = nx.from_numpy_array( adjacency_nparray )


        self._logger.info(f"[DATASET] Generating node features as per string '{self._dataset_name_string}'")
        # get node features as tensor
        nodes_list = list(G.nodes(data=False))
        all_node_features = []
        for node in nodes_list:
            temp_list = []

            # add features below:
            temp_list = self._get_sequential_node_features( self._dataset_name_string, node, adjacency_list, len(dummy_fc_fc1), model_fc_left_bias, model_fc_right_bias )

            all_node_features.append(temp_list)

        all_node_features = np.array( all_node_features, dtype = np.float64 )
        all_node_features = torch.tensor( all_node_features, dtype = torch.float64)


        self._logger.info("[DATASET] Generating edge features")
        # get edge features as tensor
        edge_weights = nx.get_edge_attributes(G, "weight")
        all_edge_features = []
        for key, value in edge_weights.items():
            temp_list = []
            temp_list.append(value)
            all_edge_features.append(temp_list)

        all_edge_features = np.array( all_edge_features, dtype = np.float64 )
        all_edge_features = torch.tensor( all_edge_features, dtype = torch.float64)


        self._logger.info("[DATASET] Generating edge list")
        # get adjacency information/edge list from linegraph
        edge_list_obj = nx.generate_edgelist(G, data = False)
        all_edge_list = []
        temp_list_x = []
        temp_list_y = []

        for line in edge_list_obj:
            line_x = int(line[0 : line.find(" ")])
            temp_list_x.append(line_x)
            line_y = int(line[line.find(" ")+1 : ])
            temp_list_y.append(line_y)

        all_edge_list.append(temp_list_x)
        all_edge_list.append(temp_list_y)

        # get edge list as tensor
        all_edge_list = np.array( all_edge_list, dtype = np.int64 )
        all_edge_list = torch.tensor( all_edge_list, dtype = torch.int64)


        self._logger.info("[DATASET] Generating labels")
        # get lables of each linegraph
        all_labels_list = [1]
        all_labels_list = np.array( all_labels_list, dtype = np.int64 )
        all_labels_list = torch.tensor( all_labels_list, dtype = torch.int64)


        self._logger.info("[DATASET] Generating data object")
        # create Data object for each linegraph in order: node features, adjacency matrix as edge list in COO format, edge features, labels
        data = Data(x = all_node_features,
                    edge_index = all_edge_list,
                    edge_attr = all_edge_features,
                    y = all_labels_list)


        self._logger.info("[DATASET] Saving data object")
        # save data in processed_dir - each graph will get it's own file in processed_dir
        torch.save(data, os.path.join(self.processed_dir, f"data_{self._dataset_name_short}_0.pt"))


    # function to get the length of the dataset
    def len(self):
        return len(self.processed_file_names)


    # function to get a single item, by item index, from the dataset
    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f"data_{self._dataset_name_short}_{idx}.pt"), map_location = self._model_device)
        return data


    # function to get the model's weights are tensor
    def _get_model_weights_as_tensor(self, model_file):
        # load model
        model = torch.load( model_file, map_location = self._model_device )
        # get model parameters list
        model_params = list( model.parameters() )
        model_params_len = len(model_params)

        # extract length-2'th set of weights - last full connected layer
        model_weights = model_params[model_params_len-2].cpu().detach().tolist()
        # extract the set of biases for the left nodes in the last fc layer
        model_fc_left_bias = model_params[model_params_len-3].cpu().detach().tolist()
        # extract the set of biases for the right nodes in the last fc layer
        model_fc_right_bias = model_params[model_params_len-1].cpu().detach().tolist()

        # convert to tensor
        model_weights = torch.tensor(model_weights)

        # next line is dummy code comment it for real models
        # model_weights = torch.tensor([[1., 2., 3.], [4., 5. , 6.]])

        return model_weights, model_fc_left_bias, model_fc_right_bias


    # function to generate dummy node ids for easy representation
    def _get_dummy_node_ids(self, model_weights):
        dummy_fc_fc1_len = len(model_weights[0])
        dummy_fc_fc1 = [i for i in range(dummy_fc_fc1_len)]

        dummy_fc_fc2_len = len(model_weights)
        dummy_fc_fc2 = [i for i in range(dummy_fc_fc1_len, dummy_fc_fc1_len + dummy_fc_fc2_len)]

        return dummy_fc_fc1, dummy_fc_fc2


    # construct adjacency matrix from the psuedo node ids and model weights
    def _get_adjacency_matrix(self, fc1_len, fc2_len, model_weights):
        max_length = fc1_len + fc2_len
        adjacency_list = [[0 for i in range(max_length)] for j in range(max_length)]

        for i in range( fc1_len ):
            for j in range( fc2_len ):
                value = model_weights[j][i]
                r = i
                c = fc1_len + j

                adjacency_list[r][c] = value
                adjacency_list[c][r] = value

        # convert adjacency matrix list to numpy array
        adjacency_nparray = np.array( adjacency_list, dtype = np.float64 )

        return adjacency_list, adjacency_nparray


    # get sequential node features
    def _get_sequential_node_features(self, dataset_name_string, node, adjacency_list, dummy_fc_fc1_len, model_fc_left_bias, model_fc_right_bias):
        temp_list = []
        feature_weights = { "1" : 1, "LR" : 1, "ND" : 1, "nodecount" : 1,
                            "mean" : 100, "min" : 100,  "max" : 100,  "sum" : 100, "bias" : 100,
                            "25histbin" : 1,  "25histboundary" : 100,
                            "histbin" : 1,  "histboundary" : 100 }
        connected_weights = adjacency_list[node]

        if("1" in dataset_name_string):
            temp_list.append( 1 * feature_weights["1"] )

        if("nodecount" in dataset_name_string):
            temp_list.append( len(adjacency_list) * feature_weights["nodecount"] )

        if("LR" in dataset_name_string):
            # 1 if the node is in left set, 2 if in right set
            if(node < dummy_fc_fc1_len):
                temp_list.append( 1 * feature_weights["LR"] )
            else:
                temp_list.append( 2 * feature_weights["LR"] )

        if("mean" in dataset_name_string):
            temp_list.append( np.mean(connected_weights) * feature_weights["mean"] )

        if("min" in dataset_name_string):
            temp_list.append( np.min(connected_weights) * feature_weights["min"] )

        if("max" in dataset_name_string):
            temp_list.append( np.max(connected_weights) * feature_weights["max"] )

        if(("25histbin" in dataset_name_string) or ("25histboundary" in dataset_name_string)):
            histbin, histboundary = np.histogram(connected_weights, bins = 25)

            if("25histbin" in dataset_name_string):
                histbin = histbin * feature_weights["25histbin"]
                temp_list.extend( histbin )
            if("25histboundary" in dataset_name_string):
                histboundary = histboundary * feature_weights["25histboundary"]
                temp_list.extend( histboundary )
        elif(("histbin" in dataset_name_string) or ("histboundary" in dataset_name_string)):
            histbin, histboundary = np.histogram(connected_weights, bins = 5)

            if("histbin" in dataset_name_string):
                histbin = histbin * feature_weights["histbin"]
                temp_list.extend( histbin )
            if("histboundary" in dataset_name_string):
                histboundary = histboundary * feature_weights["histboundary"]
                temp_list.extend( histboundary )

        if("ND" in dataset_name_string):
            # 512 if the node is in left set, 10 if in right set
            if(node < dummy_fc_fc1_len):
                temp_list.append( (len(adjacency_list)-dummy_fc_fc1_len) * feature_weights["ND"] )
            else:
                temp_list.append( dummy_fc_fc1_len * feature_weights["ND"] )

        if("sum" in dataset_name_string):
            temp_list.append( np.sum(connected_weights) * feature_weights["sum"] )

        if("bias" in dataset_name_string):
            if(node < dummy_fc_fc1_len):
                # node is part of left node group in bipartite graph
                temp_list.append( model_fc_left_bias[node] * feature_weights["bias"] )
            else:
                # node is part of right node group in bipartite graph
                temp_list.append( model_fc_right_bias[node - dummy_fc_fc1_len] * feature_weights["bias"] )

        if(self._print_example_node_feature):
            self._logger.info(f"[DATASET]   - Node feature weights: {feature_weights}")
            self._logger.info(f"[DATASET]   - Example node features list: {temp_list}")
            self._print_example_node_feature = False

        return temp_list


class GraphsDataset_Configure(Dataset):
    def __init__(self,
        scratch_dirpath: str,
        device,
        root: str,
        csv_file: str,
        dataset_name_short: str,
        dataset_name_string: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        logger = None
    ):
        # declare the necessary paths
        self._model_path = os.path.join(scratch_dirpath, "train")
        self._model_csv_path = scratch_dirpath
        self._csv_file = csv_file
        self._dataset_name_short = dataset_name_short
        self._dataset_name_string = dataset_name_string
        self._model_device = device
        self._print_example_node_feature = False
        self._logger = logger

        # copy file from model_csv_path to raw_dir path
        shutil.copy2( os.path.join(self._model_csv_path, self._csv_file), os.path.join(root, "raw/") )

        super(GraphsDataset_Configure, self).__init__(root, transform, pre_transform, pre_filter)



    @property
    def raw_file_names(self):
        return self._csv_file


    @property
    def processed_file_names(self):
        self.data = pd.read_csv(self.raw_paths[0])
        processed_files = [f"data_{self._dataset_name_short}_{index}.pt" for index in list(self.data.index)]
        return processed_files


    def download(self):
        pass


    def process(self):
        # read data from csv_file
        self.data = pd.read_csv(self.raw_paths[0])

        for index, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            row_model_file = row["model_name"]
            row_label = row["poisoned"]

            self._logger.info(f"[DATASET] ---- Converting row {index+1} {row_model_file} ----")


            self._logger.info("[DATASET] Getting model weights")
            # get cnn weights from npy file
            model_weights, model_fc_left_bias, model_fc_right_bias = self._get_model_weights_as_tensor( os.path.join(self._model_path, row_model_file) )


            self._logger.info("[DATASET] Generating dummy node ids")
            # create psuedo node ids to create graph
            dummy_fc_fc1, dummy_fc_fc2 = self._get_dummy_node_ids( model_weights )


            self._logger.info("[DATASET] Generating adjacency matrix")
            # get adjacency matrix
            adjacency_list, adjacency_nparray = self._get_adjacency_matrix( len(dummy_fc_fc1), len(dummy_fc_fc2), model_weights )


            self._logger.info("[DATASET] Generating graph G")
            # create graph from nodes weights

            G = nx.from_numpy_array( adjacency_nparray )


            self._logger.info(f"[DATASET] Generating node features as per string '{self._dataset_name_string}'")
            # get node features as tensor
            nodes_list = list(G.nodes(data=False))
            all_node_features = []
            for node in nodes_list:
                temp_list = []

                # add features below:
                temp_list = self._get_sequential_node_features( self._dataset_name_string, node, adjacency_list, len(dummy_fc_fc1), model_fc_left_bias, model_fc_right_bias )

                all_node_features.append(temp_list)

            all_node_features = np.array( all_node_features, dtype = np.float64 )
            all_node_features = torch.tensor( all_node_features, dtype = torch.float64)


            self._logger.info("[DATASET] Generating edge features")
            # get edge features as tensor
            edge_weights = nx.get_edge_attributes(G, "weight")
            all_edge_features = []
            for key, value in edge_weights.items():
                temp_list = []
                temp_list.append(value)
                all_edge_features.append(temp_list)

            all_edge_features = np.array( all_edge_features, dtype = np.float64 )
            all_edge_features = torch.tensor( all_edge_features, dtype = torch.float64)


            self._logger.info("[DATASET] Generating edge list")
            # get adjacency information/edge list from linegraph
            edge_list_obj = nx.generate_edgelist(G, data = False)
            all_edge_list = []
            temp_list_x = []
            temp_list_y = []

            for line in edge_list_obj:
                line_x = int(line[0 : line.find(" ")])
                temp_list_x.append(line_x)
                line_y = int(line[line.find(" ")+1 : ])
                temp_list_y.append(line_y)

            all_edge_list.append(temp_list_x)
            all_edge_list.append(temp_list_y)

            # get edge list as tensor
            all_edge_list = np.array( all_edge_list, dtype = np.int64 )
            all_edge_list = torch.tensor( all_edge_list, dtype = torch.int64)


            self._logger.info("[DATASET] Generating labels")
            # get lables of each linegraph
            all_labels_list = [1 if(row_label=="yes") else 0]
            all_labels_list = np.array( all_labels_list, dtype = np.int64 )
            all_labels_list = torch.tensor( all_labels_list, dtype = torch.int64)


            self._logger.info("[DATASET] Generating data object")
            # create Data object for each linegraph in order: node features, adjacency matrix as edge list in COO format, edge features, labels
            data = Data(x = all_node_features,
                        edge_index = all_edge_list,
                        edge_attr = all_edge_features,
                        y = all_labels_list)


            self._logger.info("[DATASET] Saving data object")
            # save data in processed_dir - each graph will get it's own file in processed_dir
            torch.save(data, os.path.join(self.processed_dir, f"data_{self._dataset_name_short}_{index}.pt"))


    # function to get the length of the dataset
    def len(self):
        return self.data.shape[0]


    # function to get a single item, by item index, from the dataset
    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f"data_{self._dataset_name_short}_{idx}.pt"), map_location = self._model_device)
        return data


    # function to get the model's weights are tensor
    def _get_model_weights_as_tensor(self, model_file):
        # load model
        model = torch.load( model_file, map_location = self._model_device )
        # get model parameters list
        model_params = list( model.parameters() )
        model_params_len = len(model_params)

        # extract length-2'th set of weights - last full connected layer
        model_weights = model_params[model_params_len-2].cpu().detach().tolist()
        # extract the set of biases for the left nodes in the last fc layer
        model_fc_left_bias = model_params[model_params_len-3].cpu().detach().tolist()
        # extract the set of biases for the right nodes in the last fc layer
        model_fc_right_bias = model_params[model_params_len-1].cpu().detach().tolist()

        # convert to tensor
        model_weights = torch.tensor(model_weights)

        # next line is dummy code comment it for real models
        # model_weights = torch.tensor([[1., 2., 3.], [4., 5. , 6.]])

        return model_weights, model_fc_left_bias, model_fc_right_bias


    # function to generate dummy node ids for easy representation
    def _get_dummy_node_ids(self, model_weights):
        dummy_fc_fc1_len = len(model_weights[0])
        dummy_fc_fc1 = [i for i in range(dummy_fc_fc1_len)]

        dummy_fc_fc2_len = len(model_weights)
        dummy_fc_fc2 = [i for i in range(dummy_fc_fc1_len, dummy_fc_fc1_len + dummy_fc_fc2_len)]

        return dummy_fc_fc1, dummy_fc_fc2


    # construct adjacency matrix from the psuedo node ids and model weights
    def _get_adjacency_matrix(self, fc1_len, fc2_len, model_weights):
        max_length = fc1_len + fc2_len
        adjacency_list = [[0 for i in range(max_length)] for j in range(max_length)]

        for i in range( fc1_len ):
            for j in range( fc2_len ):
                value = model_weights[j][i]
                r = i
                c = fc1_len + j

                adjacency_list[r][c] = value
                adjacency_list[c][r] = value

        # convert adjacency matrix list to numpy array
        adjacency_nparray = np.array( adjacency_list, dtype = np.float64 )

        return adjacency_list, adjacency_nparray


    # get sequential node features
    def _get_sequential_node_features(self, dataset_name_string, node, adjacency_list, dummy_fc_fc1_len, model_fc_left_bias, model_fc_right_bias):
        temp_list = []
        feature_weights = { "1" : 1, "LR" : 1, "ND" : 1, "nodecount" : 1,
                            "mean" : 100, "min" : 100,  "max" : 100,  "sum" : 100, "bias" : 100,
                            "25histbin" : 1,  "25histboundary" : 100,
                            "histbin" : 1,  "histboundary" : 100 }
        connected_weights = adjacency_list[node]

        if("1" in dataset_name_string):
            temp_list.append( 1 * feature_weights["1"] )

        if("nodecount" in dataset_name_string):
            temp_list.append( len(adjacency_list) * feature_weights["nodecount"] )

        if("LR" in dataset_name_string):
            # 1 if the node is in left set, 2 if in right set
            if(node < dummy_fc_fc1_len):
                temp_list.append( 1 * feature_weights["LR"] )
            else:
                temp_list.append( 2 * feature_weights["LR"] )

        if("mean" in dataset_name_string):
            temp_list.append( np.mean(connected_weights) * feature_weights["mean"] )

        if("min" in dataset_name_string):
            temp_list.append( np.min(connected_weights) * feature_weights["min"] )

        if("max" in dataset_name_string):
            temp_list.append( np.max(connected_weights) * feature_weights["max"] )

        if(("25histbin" in dataset_name_string) or ("25histboundary" in dataset_name_string)):
            histbin, histboundary = np.histogram(connected_weights, bins = 25)

            if("25histbin" in dataset_name_string):
                histbin = histbin * feature_weights["25histbin"]
                temp_list.extend( histbin )
            if("25histboundary" in dataset_name_string):
                histboundary = histboundary * feature_weights["25histboundary"]
                temp_list.extend( histboundary )
        elif(("histbin" in dataset_name_string) or ("histboundary" in dataset_name_string)):
            histbin, histboundary = np.histogram(connected_weights, bins = 5)

            if("histbin" in dataset_name_string):
                histbin = histbin * feature_weights["histbin"]
                temp_list.extend( histbin )
            if("histboundary" in dataset_name_string):
                histboundary = histboundary * feature_weights["histboundary"]
                temp_list.extend( histboundary )

        if("ND" in dataset_name_string):
            # 512 if the node is in left set, 10 if in right set
            if(node < dummy_fc_fc1_len):
                temp_list.append( (len(adjacency_list)-dummy_fc_fc1_len) * feature_weights["ND"] )
            else:
                temp_list.append( dummy_fc_fc1_len * feature_weights["ND"] )

        if("sum" in dataset_name_string):
            temp_list.append( np.sum(connected_weights) * feature_weights["sum"] )

        if("bias" in dataset_name_string):
            if(node < dummy_fc_fc1_len):
                # node is part of left node group in bipartite graph
                temp_list.append( model_fc_left_bias[node] * feature_weights["bias"] )
            else:
                # node is part of right node group in bipartite graph
                temp_list.append( model_fc_right_bias[node - dummy_fc_fc1_len] * feature_weights["bias"] )

        if(self._print_example_node_feature):
            self._logger.info(f"[DATASET]   - Node feature weights: {feature_weights}")
            self._logger.info(f"[DATASET]   - Example node features list: {temp_list}")
            self._print_example_node_feature = False

        return temp_list
