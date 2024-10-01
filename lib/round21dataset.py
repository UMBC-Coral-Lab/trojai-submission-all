__author__ = "Akash Vartak"

import json
import os
from pathlib import Path

import torch
import torchvision
from torchvision.transforms import v2
from PIL import Image
from torch.utils.data import Dataset


class Round21Dataset(Dataset):
    def __init__(self, root, img_exts = ["jpg", "png"], class_info_ext = "json", get_clean = True, split = "train", require_label = False, mitigation_type: str = "pruning"):
        root = Path(root)
        train_augmentation_transforms = torchvision.transforms.Compose(
            [
                v2.PILToTensor(),
                v2.RandomPhotometricDistort(),
                torchvision.transforms.RandomCrop(size = 256, padding = int(10), padding_mode = 'reflect'),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ConvertImageDtype(torch.float),
            ]
        )

        test_augmentation_transforms = torchvision.transforms.Compose(
            [
                v2.PILToTensor(),
                torchvision.transforms.ConvertImageDtype(torch.float),
            ]
        )
        if split == 'test':
            self.transform = test_augmentation_transforms
        else:
            self.transform = train_augmentation_transforms

        self.mitigation_type = mitigation_type

        self.images = []
        self.labels = []
        self.fnames = []
        self.file_counter = 1

        # training_models_dirpath = os.path.join(root, 'models')
        training_models_dirpath = root
        file_counter = 1
        # for model_dir in os.listdir(training_models_dirpath):
        # training_model_dirpath = os.path.join(training_models_dirpath, model_dir)

        print(f"Files in root: {os.listdir(training_models_dirpath)}")

        new_clean_data_example_dirpath = os.path.join(training_models_dirpath, 'new-clean-example-data')
        new_poisoned_data_example_dirpath = os.path.join(training_models_dirpath, 'new-poisoned-example-data')

        # Iterate over new poisoned example data, each example contains a filename such as '305.png', with associated ground truth '305.json'
        # Ground truth is stored as a dictionary with 'clean_label' and 'poisoned_label'
        if((not get_clean) and os.path.exists(new_poisoned_data_example_dirpath)):
            print(f"[Round21Dataset] ======= Processing {new_poisoned_data_example_dirpath} =======")
            for example_file in os.listdir(new_poisoned_data_example_dirpath):
                if example_file.endswith('.png'):
                    print(f"[Round21Dataset]   ---- File #{file_counter}: {example_file} ----")
                    example_basename_no_ext = os.path.splitext(example_file)[0]
                    example_image_filepath = os.path.join(new_poisoned_data_example_dirpath, example_file)
                    example_groundtruth_filepath = os.path.join(new_poisoned_data_example_dirpath, '{}.json'.format(example_basename_no_ext))

                    if(require_label):
                        with open(example_groundtruth_filepath, 'r') as fp:
                            example_groundtruth_dict = json.load(fp)
                            clean_label = example_groundtruth_dict['clean_label']
                            poisoned_label = example_groundtruth_dict['poisoned_label']

                            print(f"[Round21Dataset]      Image path: {example_image_filepath}")
                            print(f"[Round21Dataset]      clean_label: {clean_label}")
                            print(f"[Round21Dataset]      poisoned_label: {poisoned_label}")

                    pil_img = Image.open( example_image_filepath ).convert("RGB")
                    self.images.append( pil_img )
                    self.labels.append( clean_label )
                    self.fnames.append( Path(example_file).name )

                    file_counter += 1

        # Iterate over new clean example data, each example contains a filename such as '305.png', with associated ground truth '305.json'
        # Ground truth is stored as a dictionary with 'clean_label
        # if(split == 'test'):
        if(get_clean and os.path.exists(new_clean_data_example_dirpath)):
            print(f"[Round21Dataset] ======= Processing {new_clean_data_example_dirpath} =======")
            for example_file in os.listdir(new_clean_data_example_dirpath):
                if example_file.endswith('.png'):
                    print(f"[Round21Dataset]   ---- File #{file_counter}: {example_file} ---- ")
                    example_basename_no_ext = os.path.splitext(example_file)[0]
                    example_image_filepath = os.path.join(new_clean_data_example_dirpath, example_file)
                    example_groundtruth_filepath = os.path.join(new_clean_data_example_dirpath, '{}.json'.format(example_basename_no_ext))

                    if(require_label):
                        with open(example_groundtruth_filepath, 'r') as fp:
                            example_groundtruth_dict = json.load(fp)
                            clean_label = example_groundtruth_dict['clean_label']

                            print(f"[Round21Dataset]      Image path: {example_image_filepath}")
                            print(f"[Round21Dataset]      clean_label: {clean_label}")

                    pil_img = Image.open( example_image_filepath ).convert("RGB")
                    self.images.append( pil_img )
                    self.labels.append( clean_label )
                    self.fnames.append( Path(example_file).name )

                    file_counter += 1


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        fname = self.fnames[idx]

        img = self.transform(img)
        return img, label, fname