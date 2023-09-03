# trojai-submission-all
## Branch: round12-cyber-pdf-dec2022-svm

The SVM meta-classifier submission for TrojAI Round12 `cyber-pdf-dec2022`

Before doing anything else:
  >> conda activate toy-project-2-env


To run the GCN pipeline in configure/training mode:
  >> python gcn_pipeline.py \
        --metaparameters_filepath=./metaparameters.json  \
        --schema_filepath=./metaparameters_schema.json  \
        --scratch_dirpath=./scratch/  \
        --configure_mode \
        --configure_models_dirpath=./model/ \
        --learned_parameters_dirpath=./learned_parameters/

That will run the training process and write the trained model to a directory specified by option --learned_parameters_dirpath.


To run the GCN pipeline in inference mode on a model:
  >> python gcn_pipeline.py  \
        --metaparameters_filepath=./metaparameters.json  \
        --schema_filepath=./metaparameters_schema.json  \
        --model_filepath=./model/example_model_to_test.pt  \
        --result_filepath=./output.txt  \
        --scratch_dirpath=./scratch/  \
        --learned_parameters_dirpath=./learned_parameters/

That will run inference on the model specified by option --model_filepath and write the probability of the model being trojaned to file specified by --result_filepath


To build the container after you have trained the models:
  >> sudo singularity build  gcn_pipeline.simg  gcn_pipeline.def


To run the container in inference mode:
  >> singularity run \
        --bind /home/akash/GitHubRepos/gcn-trojai-submission-r11 \
        --nv \
        ./gcn_pipeline.simg \
        --metaparameters_filepath=./metaparameters.json  \
        --schema_filepath=./metaparameters_schema.json  \
        --model_filepath=./models_to_test/mobilenet_v2_train_model2_trojaned_single.pt \
        --result_filepath=./output.txt \
        --scratch_dirpath=./scratch/  \
        --learned_parameters_dirpath=./learned_parameters/

That will produce some warnings because the scikit versions are out of
sync, but it works.  The fix is to upgrade the python version in the
container to 3.8 from 3.7 but I'm unsure how to do that right now.


To submit the container to the test server:
  >> cp gcn_pipeline.simg image-classification-sep2022_sts_UMBC_GCN.simg
  >> gdrive upload image-classification-sep2022_sts_UMBC_GCN.simg

Replace "test" with "sts" or "holdout" to submit against those data
splits.  Or you can change the name of the dataset/round as needed.

Then go do google drive for the umbctrojai account and share that
newly upload file with trojai@nist.gov
