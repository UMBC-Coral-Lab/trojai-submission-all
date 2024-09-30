# trojai-submission-all
## Branch: round21-mitigation-image-classification-jun2024

UMBC's mitigation technique submission for Round-21 `mitigation-image-classification-jun2024` 

1. Before doing anything else: conda activate toy-project-env
 
2. Install all the requirements:
```
pip install -r requirements.txt
```

3. Install the mitigation round framework into your venv as well:
```
pip install -e ./trojai-mitigation-round-framework
```

4. If conducting mitigation, ensure you pass the mitigate flag:
```
python example_trojai_mitigation.py \
 mitigate \
 --metaparameters_filepath=metaparameters.json \
 --schema_filepath=metaparameters_schema.json \
 --model_filepath=models_to_mitigate/train-dataset/models/id-00000000/model.pt \
 --dataset_dirpath=models_to_mitigate/train-dataset/ \
 --output_dirpath=scratch/output/ \
 --model_output_name=mitigated_model.pt \
 --round_training_dataset_dirpath=models_to_mitigate/train-dataset/ \
 > logs/mitigate.txt
```

5. If conducting evaluation, you can use the test flag to separately test the cleaned model on an arbitrary dataset which produces a results.json file:
```
python3 example_trojai_mitigation.py \
 test \
 --metaparameters_filepath=metaparameters.json \
 --schema_filepath=metaparameters_schema.json \
 --model_filepath=models_to_mitigate/mitigated_model.pt \
 --dataset_dirpath=models_to_mitigate/train-dataset/ \
 --output_dirpath=scratch/output/ \ 
 > logs/mitigate_test.txt
```

6. To obtain example metrics from here, call the example_metrics.py script on the produced result.json file:
```
python3 example_metrics.py \
 --metrics f1 accuracy \
 --result_file /path/to/results.json \
 --model_name <model name to be used in csv> \
 --data_type <clean / poisoned> \
 --num_classes <class count> \ 
 > logs/mterics.txt
```

7. For debugging cuda errors, use the following as the predecessor before any `python` command:
```
CUDA_LAUNCH_BLOCKING=1
```