# trojai-submission-all
## Branch: round14-rl-lavaworld-jul2023-gcn

The GCN Pipeline submission for TrojAI for Round-14 `rl-lavaworld-jul2023`

 1. Before doing anything else: ```conda activate toy-project-rl-env```

 2. To run the GCN pipeline in one-off reconfiguration:
  ```
  python entrypoint.py configure \
          --scratch_dirpath ./scratch/ \
          --metaparameters_filepath ./metaparameters.json \
          --schema_filepath ./metaparameters_schema.json \
          --learned_parameters_dirpath ./learned_parameters/ \
          --configure_models_dirpath ./models_to_train/round14/ 
  ```

 3. To run the GCN pipeline in automatic reconfiguraiton:
  ```
  python entrypoint.py configure \
          --automatic_configuration \
          --scratch_dirpath ./scratch/ \
          --metaparameters_filepath ./metaparameters.json \
          --schema_filepath ./metaparameters_schema.json \
          --learned_parameters_dirpath ./learned_parameters/ \
          --configure_models_dirpath ./models_to_train/round14/ 
  ```

  * Any of the above options will run the training process and write the trained model to a directory specified by option `--learned_parameters_dirpath`.
  * If automatic reconfiguration is run then the last set of training parameters used will be written in file specified by `--metaparameters_filepath`.


 4. To run the GCN pipeline in inference mode on a model:
  ```
  python entrypoint.py infer \
          --model_filepath ./models_to_train/round14/id-00000000/model.pt \
          --result_filepath ./output.txt \
          --scratch_dirpath ./scratch/ \
          --examples_dirpath ./examples/ \
          --round_training_dataset_dirpath ./round_training/ \
          --metaparameters_filepath ./metaparameters.json \
          --schema_filepath ./metaparameters_schema.json \
          --learned_parameters_dirpath ./learned_parameters/ 
  ```

  * That will run inference on the model specified by option `--model_filepath` and write the probability of the model being trojaned to file specified by `--result_filepath`.


 5. To build the container after you have trained the models:
  ```
  sudo singularity build  rl-lavaworld-jul2023_sts_UMBC_gcn.simg  gcn_pipeline.def
  ```


 6. To test the created container in infer mode:
  ```
  singularity run \
          --bind /home/akash/GithubRepos/trojai-submission-all \
          --nv \
          ./rl-lavaworld-jul2023_sts_UMBC_gcn.simg \
          infer \
          --model_filepath=./models_to_train/round14/id-000000012/model.pt \
          --result_filepath=./output.txt \
          --scratch_dirpath=./scratch/ \
          --examples_dirpath=./model/id-00000119/clean-example-data/ \
          --round_training_dataset_dirpath=./round_training/ \
          --metaparameters_filepath=./metaparameters.json \
          --schema_filepath=./metaparameters_schema.json \
          --learned_parameters_dirpath=./learned_parameters/ 
  ```
  * To submit container to train or test leaderboards, change the `sts` to `train` or `test` and **rebuild** container. Rebuilding is important otherwise container does not get picked up for execution.


 7. To submit the container for execution, upload the container to GDrive of umbctrojai account and share it with trojai@nist.gov.
