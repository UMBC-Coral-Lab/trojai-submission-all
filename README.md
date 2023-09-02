# trojai-submission-all
## Branch: round12-cyber-pdf-dec2022-mlp

The MLP meta-classifier submission for TrojAI Round12 `cyber-pdf-dec2022`

 1. Before doing anything else: ```conda activate toy-project-2-env```

 2. To run the GCN pipeline in one-off reconfiguration:
  ```
  python entrypoint.py configure \
          --scratch_dirpath ./scratch/ \
          --metaparameters_filepath ./metaparameters.json \
          --schema_filepath ./metaparameters_schema.json \
          --learned_parameters_dirpath ./learned_parameters/ \
          --configure_models_dirpath ./models_to_train/ \
          --scale_parameters_filepath ./scale_params.npy
  ```

 3. To run the GCN pipeline in automatic reconfiguraiton:
  ```
  python entrypoint.py configure \
          --scratch_dirpath ./scratch/ \
          --metaparameters_filepath ./metaparameters.json \
          --schema_filepath ./metaparameters_schema.json \
          --learned_parameters_dirpath ./learned_parameters/ \
          --configure_models_dirpath ./models_to_train/ \
          --scale_parameters_filepath ./scale_params.npy \
          --automatic_configuration
  ```

  * Any of the above options will run the training process and write the trained model to a directory specified by option --learned_parameters_dirpath.
  * If automatic reconfiguration is run then the last set of training parameters used will be written in file specified by --metaparameters_filepath.


 4. To run the GCN pipeline in inference mode on a model:
  ```
  python entrypoint.py infer \
          --model_filepath ./model/id-00000119/model.pt \
          --result_filepath ./output.txt \
          --scratch_dirpath ./scratch/ \
          --examples_dirpath ./examples/ \
          --round_training_dataset_dirpath ./round_training/ \
          --metaparameters_filepath ./metaparameters.json \
          --schema_filepath ./metaparameters_schema.json \
          --learned_parameters_dirpath ./learned_parameters/ \
          --scale_parameters_filepath ./scale_params.npy
  ```

  * That will run inference on the model specified by option --model_filepath and write the probability of the model being trojaned to file specified by --result_filepath


 5. To build the container after you have trained the models:
  ```
  sudo singularity build  cyber-pdf-dec2022_sts_UMBC_MLP.simg  mlp_pipeline.def
  ```


 6. To test the created container in infer mode:
  ```
  singularity run \
          --bind /home/akash/GitHubRepos/mlp-trojai-submission-r12 \
          --nv \
          ./cyber-pdf-dec2022_sts_UMBC_MLP.simg \
          infer \
          --model_filepath=./model/id-00000119/model.pt \
          --result_filepath=./output.txt \
          --scratch_dirpath=./scratch/ \
          --examples_dirpath=./model/id-00000119/clean-example-data/ \
          --round_training_dataset_dirpath=./round_training/ \
          --metaparameters_filepath=./metaparameters.json \
          --schema_filepath=./metaparameters_schema.json \
          --learned_parameters_dirpath=./learned_parameters/ \
          --scale_parameters_filepath ./scale_params.npy
  ```
  * To submit container to train or test leaderboards, change the `sts` to `train` or `test` and **rebuild** container. Rebuilding is important otherwise container does not get picked up for execution.


 7. To submit the container for execution, upload the container to GDrive of umbctrojai account and share it with trojai@nist.gov.
