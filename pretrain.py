"""
GEM pretrain
"""

import os
from os.path import join, exists, basename
import sys
import argparse
import time
import numpy as np
from glob import glob
import logging
import paddle
import paddle.distributed as dist

from pahelix.datasets.inmemory_dataset import InMemoryDataset
from pahelix.utils import load_json_config
from pahelix.featurizers.gem_featurizer import GeoPredTransformFn, GeoPredCollateFn
from pahelix.model_zoo.gem_model import GeoGNNModel, GeoPredModel
from src.utils import exempt_parameters

def train(args, model, optimizer, data_gen):
    """
    tbd
    """
    model.trian()
    steps = get_steps_per_epoch(args)
    step = 0
    list_loss = []
    for graph_dict, feed_dict in data_gen:
        print('rank:%s step: %s' % (dist.get_rank(), step))
        for k in graph_dict:
            graph_dict[k] = graph_dict[k].tensor()
        for k in feed_dict:
            feed_dict[k] = paddle.to_tensor(feed_dict[k])
        train_loss = model(graph_dict, feed_dict)
        train_loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        list_loss.append(train_loss.numpy().mean())
        step += 1
        if step > steps:
            print("jumpping out")
            break
        return np.mean(list_loss)
    
@paddle.no_grad()
def evaluate(args, model, test_dataset, collate_fn):
    """
    tbd
    """
    model.eval()
    data_gen = test_dataset.get_data_loader(
        batch_size = args.batch_size,
        num_workers = args.num_workers,
        shuffle=True,
        collate_fn=collate_fn
    )
    dict_loss = {'loss': []}
    for graph_dict, feed_dict in data_gen:
        for k in graph_dict:
            graph_dict[k] = graph_dict[k].tensor()
        for k in feed_dict:
            feed_dict[k] = paddle.to_tensor(feed_dict[k])
        loss, sub_losses = model(graph_dict, feed_dict, return_subloss=True)

        for name in sub_losses:
            if not name in dict_loss:
                dict_loss[name] = []
            v_np = sub_losses[name].numpy()
            dict_loss[name].append(v_np)
        dict_loss['loss'] = loss.numpy()
    dict_loss = {name: np.mean(dict_loss[name]) for name in dict_loss}
    return dict_loss

def get_steps_per_epoch(args):
    if args.dataset == 'zinc':
        train_num = int(20000000 * (1 - args.test_ratio))
    else:
        raise ValueError(args.dataset)
    if args.DEBUG:
        train_num = 100
    steps_per_epoch = int(train_num / args.batch_size)
    if args.distributed:
        steps_per_epoch = int(steps_per_epoch / dist.get_world_size())
    return steps_per_epoch

def load_smiles_to_dataset(data_path):
    files = sorted(glob('%s/*' % data_path))
    data_list = []
    for file in files:
        with open(file, 'r') as f:
            tmp_data_list = [line.strip() for line in f.readlines()]
        data_list.extend(tmp_data_list)
    dataset = InMemoryDataset(data_list = data_list)
    return dataset

