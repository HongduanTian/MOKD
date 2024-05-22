"""
This code allows you to evaluate performance of a single feature extractor + pa with NCC
on the test splits of all datasets (ilsvrc_2012, omniglot, aircraft, cu_birds, dtd, quickdraw, fungi, 
vgg_flower, traffic_sign, mscoco, mnist, cifar10, cifar100). 

To test the url model on the test splits of all datasets, run:
python test_extractor_pa.py --model.name=url --model.dir ./saved_results/url

To test the url model on the test splits of ilsrvc_2012, dtd, vgg_flower, quickdraw,
comment the line 'testsets = ALL_METADATASET_NAMES' and run:
python test_extractor_pa.py --model.name=url --model.dir ./saved_results/url -data.test ilsrvc_2012 dtd vgg_flower quickdraw
"""

from multiprocessing import context
import os
import torch
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from tabulate import tabulate
from utils import check_dir, Recorder, setup_seed

from models.losses import prototype_loss, knn_loss, lr_loss, scm_loss, svm_loss
from models.model_utils import CheckPointer
from models.model_helpers import get_model
from models.pa import apply_selection, pa
from data.meta_dataset_reader import (MetaDatasetEpisodeReader, MetaDatasetBatchReader, TRAIN_METADATASET_NAMES,
                                      ALL_METADATASET_NAMES)
from config import args
tf.compat.v1.disable_eager_execution()



def main():
    # set seed
    setup_seed(seed_id=args['seed'])
    
    TEST_SIZE = 600

    # Setting up datasets
    trainsets, valsets, testsets = args['data.train'], args['data.val'], args['data.test']
    testsets = ALL_METADATASET_NAMES # comment this line to test the model on args['data.test']
    
    if args['test.mode'] == 'mdl':
        trainsets = TRAIN_METADATASET_NAMES
    elif args['test.mode'] == 'sdl':
        trainsets = ['ilsvrc_2012']
    
    test_loader = MetaDatasetEpisodeReader('test', trainsets, trainsets, testsets, test_type=args['test.type'])
    model = get_model(None, args)
    checkpointer = CheckPointer(args, model, optimizer=None)
    checkpointer.restore_model(ckpt='best', strict=False)
    model.eval()

    accs_names = ['NCC']
    train_var_accs = dict()
    var_accs = dict()

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = False
    with tf.compat.v1.Session(config=config) as session:
        # go over each test domain
        for dataset in testsets:

            if dataset in trainsets:
                lr = 0.1
            else:
                lr = 1.
                
            print(dataset)
            train_var_accs[dataset] = {name:[] for name in accs_names}
            var_accs[dataset] = {name: [] for name in accs_names}
            
            for i in tqdm(range(TEST_SIZE)):
                with torch.no_grad():
                    sample = test_loader.get_test_task(session, dataset)
                    context_features = model.embed(sample['context_images'])
                    target_features = model.embed(sample['target_images'])
                    context_labels = sample['context_labels']
                    target_labels = sample['target_labels']

                # optimize selection parameters and perform feature selection
                selection_params, train_stats= pa(context_features, context_labels, target_features, target_labels,
                                                  max_iter=40, lr=lr, distance=args['test.distance'])

                selected_context = apply_selection(context_features, selection_params)
                selected_target = apply_selection(target_features, selection_params)
                _, stats_dict, _ = prototype_loss(
                    selected_context, context_labels,
                    selected_target, target_labels, distance=args['test.distance'])

                train_var_accs[dataset]['NCC'].append(train_stats['acc'])
                var_accs[dataset]['NCC'].append(stats_dict['acc'])
            train_acc = np.array(train_var_accs[dataset]['NCC'])*100
            dataset_acc = np.array(var_accs[dataset]['NCC']) * 100
            print(f"{dataset}: train_acc {train_acc.mean():.2f}%; test_acc {dataset_acc.mean():.2f} +/- {(1.96*dataset_acc.std()) / np.sqrt(len(dataset_acc)):.2f}%")

    # Print nice results table
    print('results of {}'.format(args['experiment_name']))
    rows = []
    for dataset_name in testsets:
        row = [dataset_name]
        for model_name in accs_names:
            acc = np.array(var_accs[dataset_name][model_name]) * 100
            mean_acc = acc.mean()
            conf = (1.96 * acc.std()) / np.sqrt(len(acc))
            row.append(f"{mean_acc:0.2f} +- {conf:0.2f}")
        rows.append(row)
    out_path = os.path.join(args['out.dir'], 'weights')
    out_path = check_dir(out_path, True)
    out_path = os.path.join(out_path, '{}-{}-{}-{}-test-results.npy'.format(args['model.name'], args['test.type'], 'pa', args['test.distance']))
    np.save(out_path, {'rows': rows})

    table = tabulate(rows, headers=['model \\ data'] + accs_names, floatfmt=".2f")
    print(table)
    print("\n")


if __name__ == '__main__':
    main()




