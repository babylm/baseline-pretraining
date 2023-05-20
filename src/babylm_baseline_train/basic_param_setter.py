import argparse
import copy
import random
import pdb
import os
import os.path as osp
import sys
import json
import re
import numpy as np
import logging
import time
import torch
import functools
from tqdm import tqdm
from torch import nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torch.distributed as dist

import pt_framework.hooks.lr_updater as lr_updater
from pt_framework.dist_utils import get_dist_info, init_dist
from pt_framework.utils import mkdir_or_exist, get_root_logger, print_log
from pt_framework.hooks.hook import Hook
from pt_framework.hooks.record_saver import MongoDBSaver
from pt_framework.hooks.optimizer import OptimizerHook, DistOptimizerHook
from pt_framework.hooks.checkpoint import CkptSpecifySaveHook
import torch.optim as optimizers

from .train.env_params import MODEL_SAVE_FOLDER, USE_TPU, REC_SAVE_FOLDER

SAVE_REC_TO_FILE = os.environ.get('SAVE_REC_TO_FILE', '1')
DEBUG = os.environ.get('DEBUG', '0')
PERSISTENT_WORKERS = int(os.environ.get('PERSISTENT_WORKERS', 1))==1


def build_optimizer(model, optimizer_cfg, verbose=True):
    """Build optimizer from configs.

    Args:
        model (:obj:`nn.Module`): The model with parameters to be optimized.
        optimizer_cfg (dict): The config dict of the optimizer.
            Positional fields are:
                - type: class name of the optimizer.
                - lr: base learning rate.
            Optional fields are:
                - any arguments of the corresponding optimizer type, e.g.,
                  weight_decay, momentum, etc.
                - paramwise_options: a dict with regular expression as keys
                  to match parameter names and a dict containing options as
                  values. Options include 6 fields: lr, lr_mult, momentum,
                  momentum_mult, weight_decay, weight_decay_mult.

    Returns:
        torch.optim.Optimizer: The initialized optimizer.

    Example:
        >>> model = torch.nn.modules.Conv1d(1, 1, 1)
        >>> paramwise_options = {
        >>>     '(bn|gn)(\d+)?.(weight|bias)': dict(weight_decay_mult=0.1),
        >>>     '\Ahead.': dict(lr_mult=10, momentum=0)}
        >>> optimizer_cfg = dict(type='SGD', lr=0.01, momentum=0.9,
        >>>                      weight_decay=0.0001,
        >>>                      paramwise_options=paramwise_options)
        >>> optimizer = build_optimizer(model, optimizer_cfg)
    """
    if hasattr(model, 'module'):
        model = model.module

    optimizer_cfg = optimizer_cfg.copy()
    paramwise_options = optimizer_cfg.pop('paramwise_options', None)
    # if no paramwise option is specified, just use the global setting
    if paramwise_options is None:
        optimizer_cls = getattr(optimizers, optimizer_cfg.pop('type'))
        return optimizer_cls(params=model.parameters(), **optimizer_cfg)
    else:
        assert isinstance(paramwise_options, dict)
        params = []
        for name, param in model.named_parameters():
            param_group = {'params': [param]}
            if not param.requires_grad:
                params.append(param_group)
                continue

            for regexp, options in paramwise_options.items():
                if re.search(regexp, name):
                    for key, value in options.items():
                        if key.endswith('_mult'): # is a multiplier
                            key = key[:-5]
                            assert key in optimizer_cfg, \
                                "{} not in optimizer_cfg".format(key)
                            value = optimizer_cfg[key] * value
                        param_group[key] = value
                        if not dist.is_initialized() or dist.get_rank() == 0:
                            if verbose:
                                print_log('paramwise_options -- {}: {}={}'.format(
                                    name, key, value))

            # otherwise use the global settings
            params.append(param_group)

        optimizer_cls = getattr(optimizers, optimizer_cfg.pop('type'))
        return optimizer_cls(params, **optimizer_cfg)


def set_random_seed(seed, deterministic=False):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class SetEpochHook(Hook):
    def before_epoch(self, runner):
        dataset = runner.data_loader.dataset
        assert hasattr(dataset, 'set_epoch')
        dataset.set_epoch(runner.epoch)


class ParamsBuilder(object):
    def __init__(
            self, exp_id, col_name,
            get_dataset_func, get_model_func,  
            optimizer_cfg,
            add_train_loader_kwargs={},
            batch_processor_params=None,
            save_rec_to_file=False,
            opt_update_interval=None,
            opt_grad_clip={'max_norm': 1.0},
            opt_use_fp16=True,
            model_find_unused=False,
            database_name='babylm_train',
            desired_batch_size=None, 
            base_batch_size=16,
            max_epochs=400,
            seed=None,
            specify_iter=[50, 200, 1000, 2000, 4000],
            specify_epoch=[1, 2, 4, 8, 20, 40],
            ckpt_save_interval=50,
            shuffle=True):
        self.exp_id = exp_id
        self.params = {'max_epochs': max_epochs}
        self.col_name = col_name
        self.get_dataset_func = get_dataset_func
        self.get_model_func = get_model_func
        self.add_train_loader_kwargs = add_train_loader_kwargs
        self.batch_processor_params = batch_processor_params
        self.optimizer_cfg = optimizer_cfg
        self.save_rec_to_file = save_rec_to_file \
                or (int(SAVE_REC_TO_FILE) == 1)
        self.opt_update_interval = opt_update_interval
        self.opt_grad_clip = opt_grad_clip
        self.opt_use_fp16 = opt_use_fp16
        self.model_find_unused = model_find_unused
        self.database_name = database_name
        self.desired_batch_size = desired_batch_size
        self.base_batch_size = base_batch_size
        self.shuffle = shuffle
        self.specify_iter = specify_iter
        self.specify_epoch = specify_epoch
        self.ckpt_save_interval = ckpt_save_interval
        self.setup_opt_update_interval()
        if seed is not None:
            set_random_seed(seed)

    def get_save_params(self):
        ckpt_work_dir = os.path.join(
                MODEL_SAVE_FOLDER, self.col_name, self.exp_id)
        rec_work_dir = os.path.join(
                REC_SAVE_FOLDER, self.col_name, self.exp_id)
        save_params = {
                'cache_ckpt_keep_nums': 3,
                'ckpt_hook_builder': CkptSpecifySaveHook,
                'ckpt_hook_kwargs': {
                    'interval': self.ckpt_save_interval,
                    'out_dir': ckpt_work_dir,
                    'cache_interval': 1,
                    'specify_epoch': self.specify_epoch,
                    'specify_iter': self.specify_iter,
                    },
                }
        if self.save_rec_to_file:
            save_params['record_saver_kwargs'] = {
                    'out_dir': rec_work_dir}
        else:
            save_params['record_saver_kwargs'] = {
                    'port': 26001,
                    'database_name': self.database_name,
                    'collection_name': self.col_name,
                    'exp_id': self.exp_id,
                    'interval': 2500,
                    'by_epoch': False,
                    }
            save_params['record_saver_builder'] = MongoDBSaver
        self.params['save_params'] = save_params

        rank, _ = get_dist_info()
        if rank == 0:
            mkdir_or_exist(rec_work_dir)
            mkdir_or_exist(ckpt_work_dir)
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = osp.join(rec_work_dir, 'train_{}.log'.format(timestamp))
        logger = get_root_logger(log_file)
        self.params['logger'] = logger

    def get_num_workers_batch_size(self):
        num_workers = int(os.environ.get(
                'NUM_WORKERS', 8))
        batch_size = int(os.environ.get(
                'BATCH_SIZE', self.base_batch_size))
        rel_batch_size = float(os.environ.get('REL_BATCH_SIZE', 1.0))
        return num_workers, int(batch_size * rel_batch_size)

    def setup_opt_update_interval(self):
        if self.desired_batch_size is not None:
            assert isinstance(self.desired_batch_size, int)
            _, batch_size = self.get_num_workers_batch_size()
            _, world_size = get_dist_info()
            opt_update_interval = self.desired_batch_size // (batch_size * world_size)
            if opt_update_interval > 1:
                self.opt_update_interval = opt_update_interval

    def add_one_hook_params(self, one_hook_params):
        if 'extra_hook_params' not in self.params:
            self.params['extra_hook_params'] = []
        self.params['extra_hook_params'].append(one_hook_params)

    def add_set_epoch_hook(self):
        set_epoch_hook_params = {'builder': SetEpochHook}
        self.add_one_hook_params(set_epoch_hook_params)

    def get_train_data_params(self):
        train_data_params = {
                'dataset_builder': self.get_dataset_func,
                'shuffle': self.shuffle,
                }
        num_workers, batch_size = self.get_num_workers_batch_size()
          
        train_data_params.update({
            'batch_size': int(batch_size),
            'num_workers': int(num_workers),
            'distributed': True,
            'data_loader_kwargs': {
                'drop_last': True,
                'persistent_workers': PERSISTENT_WORKERS,
                },
            })
        train_data_params['data_loader_kwargs'].update(
                self.add_train_loader_kwargs)
        self.params['train_data_params'] = train_data_params
        if not self.shuffle:
            self.add_set_epoch_hook()

    def build_model_optimizer(
            self, get_model_func, optimizer_cfg):
        self.model = get_model_func().cuda()
        self.optimizer = build_optimizer(
                self.model, 
                optimizer_cfg)
        if self.opt_use_fp16:
            import apex
            self.model, self.optimizer = apex.amp.initialize(self.model, self.optimizer)
        self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[torch.cuda.current_device()],
                find_unused_parameters=self.model_find_unused,
                )
        return self.model, self.optimizer

    def get_model_optimizer_params(self):
        model_optimizer_params = {
                'builder': self.build_model_optimizer,
                'builder_kwargs': dict(
                    get_model_func=self.get_model_func,
                    optimizer_cfg=self.optimizer_cfg),
                }
        self.params['model_optimizer_params'] = model_optimizer_params

    def empty_func(self):
        return None

    def get_loss_params(self):
        loss_params = {}
        loss_params['builder'] = self.empty_func
        self.params['loss_params'] = loss_params

    def get_learning_rate_params(self):
        builder_name = 'Fixed'
        builder = getattr(lr_updater, builder_name + 'LrUpdaterHook')
        lr_config = dict(
            warmup='linear',
            warmup_iters=5000 if self.opt_update_interval is None\
                         else 5000 * self.opt_update_interval,
            warmup_ratio=0.0001, # cannot be 0
            warmup_by_epoch=False)
        learning_rate_params = {
                'builder': builder,
                'builder_kwargs': lr_config,
                }
        self.params['learning_rate_params'] = learning_rate_params

    def naive_processor(self, model, loss_func, data_batch):
        if not USE_TPU:
            model_outputs = model(**data_batch)
            return {'loss': model_outputs['loss']}
        else:
            model_outputs = model(
                    return_dict=False, **data_batch)
            return {'loss': model_outputs[0]}

    def get_batch_processor_params(self):
        if self.batch_processor_params is None:
            batch_processor_params = {
                    'func': self.naive_processor,
                    }
        else:
            batch_processor_params = self.batch_processor_params
        self.params['batch_processor_params'] = batch_processor_params

    def get_validation_params(self):
        self.params['validation_params'] = {}

    def get_optimizer_hook_params(self):
        optimizer_hook_params = {
                'builder': OptimizerHook,
                'builder_kwargs': {
                    'grad_clip': self.opt_grad_clip,
                    }}
        if self.opt_update_interval is not None:
            assert isinstance(self.opt_update_interval, int)
            optimizer_hook_params['builder'] = DistOptimizerHook
            optimizer_hook_params['builder_kwargs'].update({
                        'update_interval': self.opt_update_interval})
        if self.opt_use_fp16:
            optimizer_hook_params['builder'] = DistOptimizerHook
            optimizer_hook_params['builder_kwargs'].update({
                        'use_fp16': self.opt_use_fp16})
        self.params['optimizer_hook_params'] = optimizer_hook_params

    def build_params(self):
        self.get_save_params()
        self.get_train_data_params()
        self.get_model_optimizer_params()
        self.get_loss_params()
        self.get_learning_rate_params()
        self.get_batch_processor_params()
        self.get_validation_params()
        self.get_optimizer_hook_params()
        return self.params
