__author__ = "Akash Vartak"

from typing import Dict
from pathlib import Path
import time

import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from tqdm import tqdm
import numpy as np

from trojai_mitigation_round.mitigations.mitigation import TrojAIMitigation
from trojai_mitigation_round.mitigations.mitigated_model import TrojAIMitigatedModel


class PruningMitigationTrojai(TrojAIMitigation):
    def __init__(self, device, loss_cls, optim_cls, lr, epochs, ckpt_dir = "./ckpts", ckpt_every = 0, batch_size = 32, num_workers = 1, **kwargs):
        super().__init__(device, batch_size, num_workers, **kwargs)
        self._optimizer_class = optim_cls
        self._loss_cls = loss_cls
        self.lr = lr
        self.epochs = epochs
        self.ckpt_dir = ckpt_dir
        self.ckpt_every = ckpt_every


    def get_model_architecture(self, model):
        """
        Function to get the model architecture and final layer index
        """
        model_arch = None

        # get model architecture class name
        model_arch = str(model.__class__.__name__)

        # List parameter sizes of both models
        model_params = list(model.parameters())
        last_fc_layer_idx = len(model_params) - 2
        incoming_nodes_count, outgoing_nodes_count = model_params[last_fc_layer_idx].shape[1],model_params[last_fc_layer_idx].shape[0]

        print(f"Model type: {model_arch}")

        return model_arch, incoming_nodes_count


    def zero_weights(self, model: torch.nn.Module, model_arch: str, node_idx: int):
        """
        Function to set the weights of node 'node_idx' in the fc-layer to zero
        :param model:
        :return:
        """
        print(f"[ZEROING] Zeroing node #{node_idx}")

        original_fc_weights, copy_original_fc_weights = None, None
        if(model_arch in "ResNet"):
            original_fc_weights = model.fc.weight
            copy_original_fc_weights = model.fc.weight
        elif(model_arch in "VisionTransformer"):
            original_fc_weights = model.head.weight
            copy_original_fc_weights = model.head.weight
        elif(model_arch in "MobileNetV2"):
            original_fc_weights = model.classifier[1].weight
            copy_original_fc_weights = model.classifier[1].weight

        original_fc_weights = original_fc_weights.detach().cpu().numpy()
        original_fc_weights[node_idx][:] = 0

        if(model_arch in "ResNet"):
            model.fc.weight = torch.nn.Parameter(torch.from_numpy( original_fc_weights ))
        elif(model_arch in "VisionTransformer"):
            model.head.weight = torch.nn.Parameter(torch.from_numpy( original_fc_weights ))
        elif(model_arch in "MobileNetV2"):
            model.classifier[1].weight = torch.nn.Parameter(torch.from_numpy( original_fc_weights ))

        return model

    def custom_collate(self, batch):
        """
        Custom collate function to manage batches
        """
        image = torch.stack([item[0] for item in batch], 0)
        target = torch.LongTensor([item[1] for item in batch])
        fname = [item[2] for item in batch]
        return [image, target, fname]
    
    def get_samplers(self, data_size, train_size: float = 0.8):
        """
        Generate random subset samplers given the size of the data and the split.
        """
        num_tr = int(data_size * train_size)

        print(f"Train split: [0 : {num_tr})")
        print(f"Test split: [{num_tr + 1} : {data_size})")

        # Indices.
        arr = np.array(range(data_size))
        np.random.shuffle(arr) # Shuffle the array.
        tr_idx = arr[0 : num_tr]
        te_idx = arr[num_tr : ]

        return SubsetRandomSampler(tr_idx), SubsetRandomSampler(te_idx)


    def mitigate_model(self, model: torch.nn.Module, clean_dataset: Dataset, poisoned_dataset: Dataset) -> TrojAIMitigatedModel:
        """
        Args:
            model: the model to repair
            dataset: a dataset of examples
        Returns:
            mitigated_model: A TrojAIMitigatedModel object corresponding to new model weights and a pre/post processing techniques
        """
        print("Mitigating model in PruningMitigationTrojai")
        model_arch, node_count = self.get_model_architecture(model)

        all_kl_div = {}
        # node_count = 5
        for node_idx in range(node_count):
            print(f"================== Node: {node_idx} ==================")
            # We have original 'model' and zeroed 'mitigated_model'
            mitigated_model = self.zero_weights(model, model_arch, node_idx)

            # run both models in eval mode and compare kl divergence results
            model.eval()
            mitigated_model.eval()

            if(clean_dataset is not None):
                clean_dataloader = DataLoader(clean_dataset,
                                        batch_size = self.batch_size,
                                        shuffle = True,
                                        # sampler = tr_sampler,
                                        collate_fn = self.custom_collate,
                                        num_workers = self.num_workers,
                                        drop_last = True,
                                        pin_memory = True)
            if(poisoned_dataset is not None):
                poisoned_dataloader = DataLoader(poisoned_dataset,
                                        batch_size = self.batch_size,
                                        shuffle = True,
                                        # sampler = tr_sampler,
                                        collate_fn = self.custom_collate,
                                        num_workers = self.num_workers,
                                        drop_last = True,
                                        pin_memory = True)

            # clean_model_probs = []
            # clean_mitigated_model_probs = []
            clean_kl_div = 0

            if(clean_dataset is not None):
                clean_evaldata = tqdm(enumerate(clean_dataloader), unit = " batch")
                print(f"   [COMPARE MODELS (Clean images) Node: {node_idx}] Total number of batches: {len(clean_dataloader)}")
                for batch_id, (image, label, _) in clean_evaldata:
                    # print(f"      Images in batch: {len(image)}")

                    image = image.to(self.device)

                    # For each mode, we:
                    # 1. Move model to gpu device
                    # 2. Get logits and then get class probabilities
                    # 3. Detach logits. This is crucial, to be able to remove model graph from gpu memory and effectively reclaim memory
                    # 4. Move model back to cpu
                    # 5. Empty cache to free up gpu memory. This step is where we reclaim gpu mem
                    model = model.to(self.device)
                    output = model(image)
                    pred_original = torch.log_softmax(output.detach(), dim = 1)
                    # print(f"      output.detach() {output.detach()}")
                    # print(f"      output.detach().shape: {output.detach().shape}")
                    # print(f"      pred_original: {pred_original}")
                    # print(f"      pred_original.shape: {pred_original.shape}")
                    # print(f"      torch.sum(pred_original): {torch.sum(pred_original, 1)}")
                    # print(f"      torch.sum(pred_original).shape: {torch.sum(pred_original, 1).shape}")
                    model = model.to('cpu')
                    torch.cuda.empty_cache()

                    mitigated_model = mitigated_model.to(self.device)
                    output = mitigated_model(image)
                    pred_mitigated = torch.softmax(output.detach(), dim = 1)
                    # print(f"      output.detach() {output.detach()}")
                    # print(f"      output.detach().shape: {output.detach().shape}")
                    # print(f"      pred_mitigated: {pred_mitigated}")
                    # print(f"      pred_mitigated.shape: {pred_mitigated.shape}")
                    # print(f"      torch.sum(pred_mitigated): {torch.sum(pred_mitigated, 1)}")
                    # print(f"      torch.sum(pred_mitigated).shape: {torch.sum(pred_mitigated, 1).shape}")
                    mitigated_model = mitigated_model.to('cpu')
                    torch.cuda.empty_cache()

                    # clean_model_probs.extend(pred_original)
                    # clean_mitigated_model_probs.extend(pred_mitigated)

                    # print(f"      len(model_probs): {len(model_probs)}")
                    # print(f"      len(model_probs[0]): {len(model_probs[0])}")
                    # print(f"      len(mitigated_model_probs): {len(mitigated_model_probs)}")
                    # print(f"      len(mitigated_model_probs[0]): {len(mitigated_model_probs[0])}")

                    # the expected excess surprise from using 'model' as a model instead of 'mitigated_model'
                    clean_kl_div += torch.nn.functional.kl_div(pred_original, pred_mitigated, reduction = 'sum')
                    clean_evaldata.set_description(f"[COMPARE MODELS (Clean images) Node: {node_idx}] Batch: {batch_id} | KL Divergence: {clean_kl_div}")

            # poisoned_model_probs = []
            # poisoned_mitigated_model_probs = []
            poisoned_kl_div = 0

            if(poisoned_dataloader is not None and len(poisoned_dataloader) > 0):
                poisoned_evaldata = tqdm(enumerate(poisoned_dataloader), unit = " batch")
                print(f"   [COMPARE MODELS (Posioned images) Node: {node_idx}] Total number of batches: {len(poisoned_dataloader)}")
                for batch_id, (image, label, _) in poisoned_evaldata:
                    # print(f"      Images in batch: {len(image)}")

                    image = image.to(self.device)

                    # For each mode, we:
                    # 1. Move model to gpu device
                    # 2. Get logits and then get class probabilities
                    # 3. Detach logits. This is crucial, to be able to remove model graph from gpu memory and effectively reclaim memory
                    # 4. Move model back to cpu
                    # 5. Empty cache to free up gpu memory. This step is where we reclaim gpu mem
                    model = model.to(self.device)
                    output = model(image)
                    pred_original = torch.log_softmax(output.detach(), dim = 1)
                    model = model.to('cpu')
                    torch.cuda.empty_cache()

                    mitigated_model = mitigated_model.to(self.device)
                    output = mitigated_model(image)
                    pred_mitigated = torch.softmax(output.detach(), dim = 1)
                    mitigated_model = mitigated_model.to('cpu')
                    torch.cuda.empty_cache()

                    # poisoned_model_probs.extend(pred_original)
                    # poisoned_mitigated_model_probs.extend(pred_mitigated)

                    # the expected excess surprise from using 'model' as a model instead of 'mitigated_model'
                    poisoned_kl_div += torch.nn.functional.kl_div(pred_original, pred_mitigated, reduction = 'sum')
                    poisoned_evaldata.set_description(f"[COMPARE MODELS (Poisoned images) Node: {node_idx}] Batch: {batch_id} | KL Divergence: {poisoned_kl_div}")



            all_kl_div[node_idx] = poisoned_kl_div - clean_kl_div

            print(f"   [COMPARE MODELS Node: {node_idx}] Clean KL Divergence: {clean_kl_div}")
            print(f"   [COMPARE MODELS Node: {node_idx}] Posioned KL Divergence: {poisoned_kl_div}")
            print(f"   [COMPARE MODELS Node: {node_idx}] Overall KL Divergence: {poisoned_kl_div - clean_kl_div}")

            # remove mitigated_model from gpu and delete to free up space and memory clean up
            import gc
            mitigated_model.to('cpu')
            del mitigated_model
            gc.collect()
            torch.cuda.empty_cache()

        # get best KL divergence model
        best_kl_div = 0
        best_node_idx = 0
        for node_idx, kl_div in all_kl_div.items():
            print(f"[GET BEST] Node {node_idx} KL Divergence: {kl_div}")
            if(kl_div > best_kl_div):
                best_kl_div = kl_div
                best_node_idx = node_idx
        print(f"[GET BEST] Best mitigated model has node #{best_node_idx} pruned with KL Divergence: {best_kl_div}")
        mitigated_model = self.zero_weights(model, model_arch, best_node_idx)

        return TrojAIMitigatedModel(mitigated_model)