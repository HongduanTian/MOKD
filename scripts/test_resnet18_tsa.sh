#! /bin/bash
ulimit -n 50000
export META_DATASET_ROOT=#PATH/TO/META-DATASET/CODE/FILE
export RECORDS=#PATH/TO/DATATSET
CUDA_VISIBLE_DEVICES=<gpu_id> python test_extractor_tsa.py --model.name=url --model.dir ./url --test.tsa-ad-type residual --test.tsa-ad-form matrix --test.tsa-opt alpha+beta --test.tsa-init eye --test.mode mdl --seed=42