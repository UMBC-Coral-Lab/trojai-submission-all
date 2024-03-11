from __future__ import print_function


__author__ = "Akash Vartak"
__email__ = "akashvartak@umbc.edu"

# Standard library imports
import csv
import glob
import json
import os
import shutil
import pandas as pd

import torch
from lib.gcn_model import load_drebinn_model


def copy_all_models(configure_models_dirpath, scratch_dirpath, logger):
    src_models_path = configure_models_dirpath
    source_model_folders = glob.glob( src_models_path + "id-0*/")

    dest_models_path = os.path.join(scratch_dirpath, "train/")
    if not os.path.exists( dest_models_path ):
        os.makedirs( dest_models_path )

    csv_file = "all_models.csv"
    csv_path = os.path.join(scratch_dirpath, csv_file)
    open( csv_path, 'w').close()
    outfile = open( csv_path, 'a' )
    outfileWriter = csv.writer( outfile, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_ALL )
    outfileWriter.writerow(["model_name", "poisoned"])

    logger.info(f"[COPY] Copying models from: '{src_models_path}'")
    logger.info(f"[COPY] Copying models to: '{dest_models_path}'")
    logger.info(f"[COPY] Creating csv of all copied models: '{csv_path}'")

    models_counter = 1
    for folder in source_model_folders:
        logger.info(f"[COPY] -- Opening folder: '{folder}' --")

        # model.pt - model file
        # ground_truth.csv - whether model was poisoned or not

        dataFrame = pd.read_csv(folder + "ground_truth.csv", header = None)
        poisoned = dataFrame[0][0] == 1

        if(poisoned):
            poisoned = "yes"
            new_model_file_name = f"rl_round14_model{models_counter}_trojaned.pt"
        else:
            poisoned = "no"
            new_model_file_name = f"rl_round14_model{models_counter}.pt"

        model_new = load_drebinn_model(os.path.join(folder, "model.pt"))
        torch.save(model_new, os.path.join(dest_models_path, new_model_file_name))

        # logger.info(f"[COPY]   Model poisoned: {poisoned}")
        logger.info(f"[COPY]   -> Model copied from: {os.path.join(folder, 'model.pt')}")
        logger.info(f"[COPY]   -> Model copied to: {os.path.join(dest_models_path, new_model_file_name)}")

        models_counter += 1

        outfileWriter.writerow([new_model_file_name, poisoned])

    logger.info("[COPY] Copying done.")

    return dest_models_path, csv_file
