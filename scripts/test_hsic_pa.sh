#! /bin/bash
ulimit -n 50000
export META_DATASET_ROOT=#PATH/TO/META-DATASET/CODE/FILE
export RECORDS=#PATH/TO/DATATSET
CUDA_VISIBLE_DEVICES=<gpu_id> python hsic_loss.py  --model.name=url --model.dir ./url --data.imgsize=84 \
                                            --seed=41 \
                                            --test_size=600 \
                                            --kernel.type=rbf \
                                            --epsilon=1e-5 \
                                            --test.type=standard \
                                            --experiment_name=mokd_seed41