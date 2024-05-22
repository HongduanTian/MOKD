from multiprocessing import context
import sys
import os
import torch
import torch.nn.functional as F
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from tabulate import tabulate
from utils import check_dir, device

from models.hsic_estimation import HSICEstimator
from models.model_utils import CheckPointer
from models.model_helpers import get_model
from models.pa import apply_selection
from data.meta_dataset_reader import (MetaDatasetEpisodeReader, TRAIN_METADATASET_NAMES, ALL_METADATASET_NAMES)
from config import args, GAMMA_VARY, GAMMA_5SHOT, GAMMA_1SHOT
tf.compat.v1.disable_eager_execution()


EPSILON = args['epsilon']
IS_SCALED = True
SAVE_ROOT = '/tmp/'


def compute_prototypes(embeddings:torch.Tensor, labels:torch.Tensor):
    '''
    Args:
        embeddings: [n_embeddings, c, h, w]
        labels: [n_embeddings, ]
    Return:
        prototypes: Tensor with size [num_classes, c, h, w]
    '''
    unique_labels = torch.range(start=0, end=torch.max(labels)).unsqueeze(dim=1).type_as(labels)    # [n_cls, 1]
    matrix = unique_labels.eq(labels.reshape(1, list(labels.shape)[0])).type_as(embeddings)
    prototypes = torch.matmul(matrix, embeddings) / matrix.sum(dim=1, keepdim=True)
    
    return prototypes


def pred_with_protos(proto_data, proto_labels, pred_data, pred_labels):
    '''
    Calculating the predictions in the way of Prototypical Nets
    Args:
        proto_data: Tensor with size [num_support, c, h, w];
        proto_labels: Tensor with size [num_support, 1];
        pred_data: Tensor with size [num_query, c, h, w];
        pred_labels: Tensor with size [num_query, 1]
    Return:
        acc: A scalar tensor.
    '''
    prototypes = compute_prototypes(proto_data, proto_labels)
    logits = F.cosine_similarity(pred_data.flatten(1).unsqueeze(1),
                                 prototypes.flatten(1).unsqueeze(0),
                                 dim=-1,
                                 eps=1e-30)*10
    log_p_y = F.log_softmax(logits, dim=1)
    preds = log_p_y.argmax(1)
    pred_labels = pred_labels.type(torch.long)
    acc = torch.eq(preds, pred_labels).float().mean()
    
    return acc


def hsic_loss_fn(X:torch.tensor, Y:torch.tensor, 
                 scale_hzy:torch.tensor, scale_hzz:torch.tensor, 
                 esitmation_fn, gamma):
    
    hzy = esitmation_fn.estimate_unbiased_hsic(F.normalize(X.flatten(1), dim=-1),
                                               F.normalize(F.one_hot(Y).flatten(1).float(), dim=-1),
                                               scale=scale_hzy, stat_mode=args['stat.type'], is_scaled=IS_SCALED)
    
    hzz = esitmation_fn.estimate_unbiased_hsic(F.normalize(X.flatten(1), dim=-1),
                                               F.normalize(X.flatten(1), dim=-1),
                                               scale=scale_hzz, stat_mode=args['stat.type'], is_scaled=IS_SCALED)
    
    return -hzy + gamma*hzz


def hsic_pa(context_features, context_labels, target_features, target_labels, hsicestimator, max_iter, lr, gamma, weight_decay):

    # generate parameters
    input_dim = context_features.size(1)
    output_dim = input_dim
    params = [torch.eye(output_dim, input_dim).unsqueeze(-1).unsqueeze(-1).to(device).requires_grad_(True)]

    # optimizer
    optimizer = torch.optim.Adadelta(params, lr=lr, weight_decay=weight_decay)
    
    # bandwidth choices
    if args['kernel.type'] in ['rbf', 'imq']:
        bestScale_hzy, bestScale_hzz = hsicestimator.get_best_bandwidths(context_features, context_labels, stat_mode=args['stat.type'])
    else:
        bestScale_hzy, bestScale_hzz = None, None

    # running data collection
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    for i in range(max_iter):

        with torch.no_grad():
            transformed_context = apply_selection(context_features, params)
            transformed_target = apply_selection(target_features, params)
            val_acc = pred_with_protos(transformed_context, context_labels, transformed_target, target_labels)
            val_loss = hsic_loss_fn(transformed_target, target_labels, 
                                    scale_hzy=bestScale_hzy, scale_hzz=bestScale_hzz, 
                                    esitmation_fn=hsicestimator, gamma=gamma)
            val_losses.append(val_loss.item())
            val_accs.append(val_acc.item())
        
        optimizer.zero_grad()
        transformed_context = apply_selection(context_features, params)
        train_acc = pred_with_protos(transformed_context, context_labels, transformed_context, context_labels)
        train_loss = hsic_loss_fn(transformed_context, context_labels, 
                                  scale_hzy=bestScale_hzy, scale_hzz=bestScale_hzz, 
                                  esitmation_fn=hsicestimator, gamma=gamma)
        train_stat = {'loss': train_loss.item(), 'acc': train_acc.item()}

        train_losses.append(train_loss.item())
        train_accs.append(train_acc.item())

        train_loss.backward()
        optimizer.step()
        
        if i == max_iter - 1:
            with torch.no_grad():
                transformed_context = apply_selection(context_features, params)
                transformed_target = apply_selection(target_features, params)
                val_acc = pred_with_protos(transformed_context, context_labels, transformed_target, target_labels)
                val_loss = hsic_loss_fn(transformed_target, target_labels, 
                                        scale_hzy=bestScale_hzy, scale_hzz=bestScale_hzz, 
                                        esitmation_fn=hsicestimator, gamma=gamma)
                val_stat = {'loss': val_loss.item(), 'acc': val_acc.item()}
                val_losses.append(val_loss.item())
                val_accs.append(val_acc.item())

    value_dict = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs
    }
    return train_stat, val_stat, value_dict
            

def main():

    # Setting up datasets
    trainsets, valsets, testsets = args['data.train'], args['data.val'], args['data.test']
    testsets = ALL_METADATASET_NAMES # comment this line to test the model on args['data.test']
    
    if args['test.mode'] == 'mdl':
        trainsets = TRAIN_METADATASET_NAMES
    elif args['test.mode'] == 'sdl':
        trainsets = ['ilsvrc_2012']

    test_loader = MetaDatasetEpisodeReader('test', trainsets, trainsets, testsets, test_type=args['test.type'])
    
    # Setting up model & Records
    model = get_model(None, args)
    checkpointer = CheckPointer(args, model, optimizer=None)
    checkpointer.restore_model(ckpt='best', strict=False)
    model.eval()
    
    # Initialize hsic estimator
    hsicEstimator = HSICEstimator()

    from utils import Recorder
    datarecorder = Recorder(saveroot=SAVE_ROOT, 
                            datasets=testsets, 
                            key_wd_list=['train_losses', 'train_accs', 'val_losses', 'val_accs'])
    

    accs_names = ['NCC']
    train_var_accs = dict()
    var_accs = dict()

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = False
    with tf.compat.v1.Session(config=config) as session:
        # go over each test domain
        for dataset in testsets:
            # set learning rate
            if dataset in ['traffic_sign', 'mnist']:
                lr = 1.0
            else:
                lr = 0.25
            
            if dataset in TRAIN_METADATASET_NAMES:
                weight_decay = 0.25
            else:
                weight_decay = 0.0
            
            # set gamma
            if args['test.type'] == 'standard':
                gamma = GAMMA_VARY
            elif args['test.type'] == '5shot':
                gamma = GAMMA_5SHOT
            elif args['test.type'] == '1shot':
                gamma = GAMMA_1SHOT
            else:
                raise ValueError("Unrecognized task configurations!")

            print(dataset)
            train_var_accs[dataset] = {name:[] for name in accs_names}
            var_accs[dataset] = {name: [] for name in accs_names}

            for i in tqdm(range(args['test_size'])):
                with torch.no_grad():
                    sample = test_loader.get_test_task(session, dataset)
                    context_features = model.embed(sample['context_images'])
                    target_features = model.embed(sample['target_images'])
                    context_labels = sample['context_labels']
                    target_labels = sample['target_labels']

                # optimize selection parameters and perform feature selection
                train_stats, val_stat, value_dict = hsic_pa(context_features, context_labels, 
                                                target_features, target_labels,
                                                hsicestimator=hsicEstimator,
                                                max_iter=40, lr=lr, gamma=gamma[dataset], weight_decay=weight_decay)

                datarecorder.update_records(dataset=dataset, valueDict=value_dict)  # once only one record (40 iter)

                train_var_accs[dataset]['NCC'].append(train_stats['acc'])
                var_accs[dataset]['NCC'].append(val_stat['acc'])
            train_acc = np.array(train_var_accs[dataset]['NCC'])*100
            dataset_acc = np.array(var_accs[dataset]['NCC']) * 100
            print(f"{dataset}: train_acc {train_acc.mean():.2f}%; test_acc {dataset_acc.mean():.2f} +/- {(1.96*dataset_acc.std()) / np.sqrt(len(dataset_acc)):.2f}%")

        datarecorder.save(filename=args['experiment_name'])

    # Print nice results table
    print('results of {}'.format(args['experiment_name']))
    res = {}
    rows = []
    for dataset_name in testsets:
        row = [dataset_name]
        for model_name in accs_names:
            acc = np.array(var_accs[dataset_name][model_name]) * 100
            mean_acc = acc.mean()
            conf = (1.96 * acc.std()) / np.sqrt(len(acc))
            row.append(f"{mean_acc:0.2f} +- {conf:0.2f}")
        rows.append(row)
        res[dataset_name] = [np.round(mean_acc, 2), np.round(conf, 2)]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
    import pandas as pd
    df = pd.DataFrame(res)
    
    res_path = './exp_res/'
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    df.to_csv(os.path.join(res_path, args['experiment_name']+'.csv'), index=False)
    
    out_path = os.path.join(args['out.dir'], 'weights')
    out_path = check_dir(out_path, True)
    out_path = os.path.join(out_path, '{}-{}-{}-{}-test-results.npy'.format(args['model.name'], args['test.type'], 'pa', args['test.distance']))
    np.save(out_path, {'rows': rows})

    table = tabulate(rows, headers=['model \\ data'] + accs_names, floatfmt=".2f")
    print(table)
    print("\n")


if __name__ == '__main__':
    main()
