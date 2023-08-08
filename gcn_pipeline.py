from __future__ import print_function


__author__ = "Akash Vartak"
__email__ = "akashvartak@umbc.edu"

# Standard library imports
import json
import jsonschema
import logging
import sys
import warnings
from jsonargparse import ActionConfigFile, ArgumentParser

from run_gcn_model import configure_mode, inference_mode


warnings.filterwarnings('ignore')


if(__name__ == "__main__"):
    parser = ArgumentParser(description = "Graph Convolutional Network to Detect Trojaned Models for Round-11 (image-classification-sep2022)")

    parser.add_argument("--model_filepath"
                        , type = str, default = None)
    parser.add_argument("--source_dataset_dirpath"
                        , type = str, default = None)
    parser.add_argument("--features_filepath"
                        , type = str, default = None)
    parser.add_argument("--result_filepath"
                        , type = str, default = "./output.txt")
    parser.add_argument("--scratch_dirpath"
                        , type = str, default = None)
    parser.add_argument("--examples_dirpath"
                        , type = str, default = None)
    parser.add_argument("--round_training_dataset_dirpath"
                        , type = str, default = None)
    parser.add_argument("--metaparameters_filepath"
                        , action = ActionConfigFile)
    parser.add_argument("--schema_filepath"
                        , type = str, default = None)
    parser.add_argument("--learned_parameters_dirpath"
                        , type = str, default = "learned_parameters")
    parser.add_argument("--configure_mode"
                        , default = False, action = "store_true")
    parser.add_argument("--configure_models_dirpath"
                        , type = str)

    # these parameters need to be defined here, but their values will be loaded from the json file instead of the command line
    parser.add_argument("--epochs", type = int
                        , help = "Number of epochs to train")
    parser.add_argument("--batch_size", type = int
                        , help = "Batch size for data")
    parser.add_argument("--step_size", type = int
                        , help = "Number of epochs to step for LR decay.")

    args = parser.parse_args()

    logger = logging.getLogger('logger-gcn_pipeline')
    logging.basicConfig(level = logging.INFO, stream = sys.stdout,
                        format = "%(asctime)s [%(levelname)s] %(message)s")
    logger.info("[WRAPPER] %s launched" % sys.argv[0])
    logger.info("[WRAPPER] Arguments\n" + "\n".join(["%s = %s" % (key, args[key]) for key in args.keys()]))

    # Validate config file against schema
    config_json = None
    if(args.metaparameters_filepath is not None):
        with open(args.metaparameters_filepath[0]()) as config_file:
            config_json = json.load(config_file)

            if(args.epochs is None):
                args.epochs = config_json["epochs"]
            if(args.batch_size is None):
                args.batch_size = config_json["batch_size"]
            if(args.step_size is None):
                args.step_size = config_json["step_size"]

    if(args.schema_filepath is not None):
        with open(args.schema_filepath) as schema_file:
            schema_json = json.load(schema_file)

        # this throws a fairly descriptive error if validation fails
        jsonschema.validate(instance = config_json, schema = schema_json)

    logger.info(args)

    # Start pipeline run in either configure mode or inference mode
    if(args.configure_mode):
        logger.info("[WRAPPER] Running in configure mode.")
        if( (args.configure_models_dirpath is None) or
            (args.learned_parameters_dirpath is None) ):
            logging.error("[WRAPPER] Missing parameters needed for configure mode")
            sys.exit(0)

        configure_mode( args.configure_models_dirpath,
                        args.learned_parameters_dirpath,
                        args.scratch_dirpath,
                        args.epochs, args.batch_size, args.step_size, logger)

        logger.info("[WRAPPER]  ========= Done training =========")

    else:
        logger.info("[WRAPPER] Running in inference mode.")
        if((args.model_filepath is None) or
            (args.result_filepath is None) or
            (args.learned_parameters_dirpath is None) ):
            logging.error("[WRAPPER] Missing parameters needed for inference mode")
            sys.exit(0)

        inference_mode( args.model_filepath,
                        args.result_filepath,
                        args.learned_parameters_dirpath,
                        args.scratch_dirpath,
                        args.epochs, args.batch_size, args.step_size, logger )

        logger.info("[WRAPPER]  ========= Done with inference=========")
