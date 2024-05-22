#! /bin/bash
ulimit -n 50000
export META_DATASET_ROOT=#PATH/TO/META-DATASET/CODE/FILE
export RECORDS=#PATH/TO/DATATSET
CUDA_VISIBLE_DEVICES=<gpu_id> python test_extractor_pa.py --model.name=url --model.dir ./url \
                            --test.type=standard \
                            --seed=42 \
                            --experiment_name=url_baseline_seed42
