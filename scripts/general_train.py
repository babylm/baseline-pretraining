import argparse
import os
import pdb
import importlib
import setuptools
import functools

import warnings
warnings.filterwarnings("ignore")

from pt_framework.epoch_based_runner import EpochBasedRunner
from pt_framework.dist_utils import init_dist


from babylm_baseline_train.basic_param_setter import ParamsBuilder
import babylm_baseline_train.models.helper as helper
from babylm_baseline_train.train.tk_funcs import\
        get_tokenizer_func
from babylm_baseline_train.datasets.babyLM import get_babyLM_10M
from babylm_baseline_train.datasets.utils import collate_fn
from babylm_baseline_train.train.utils import get_setting_func


def get_parser():
    parser = argparse.ArgumentParser(
            description='Pytorch training framework for general dist training')
    parser.add_argument(
            '--setting', 
            default=None, type=str, 
            action='store')
    parser.add_argument(
            '--local_rank', type=int, default=0,
            help='Used during distributed training')
    parser.add_argument(
            '--train_upto_epoch', type=int, default=None,
            help='Number of epochs to be run upto')
    parser.add_argument(
            '--exp_id', 
            default='test_strict_small', type=str, 
            action='store')
    parser.add_argument(
            '--opt_model_size', 
            default='125m', type=str, 
            action='store')
    return parser


def get_key_params(args):
    get_model_func = functools.partial(
            helper.get_opt_func, 
            opt_model_size=args.opt_model_size)

    tokenizer = get_tokenizer_func()
    add_train_loader_kwargs = dict(collate_fn=collate_fn)

    params = dict(
            exp_id=args.exp_id, col_name='babylm_test',
            get_model_func=get_model_func,
            get_dataset_func=functools.partial(
                get_babyLM_10M, tokenizer=tokenizer),
            optimizer_cfg=dict(
                type='AdamW', lr=1e-4, weight_decay=0.1,
                ),
            add_train_loader_kwargs=add_train_loader_kwargs,
            desired_batch_size=128,
            base_batch_size=128,
            )
    return params


def main():
    parser = get_parser()
    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    init_dist('pytorch')
    
    key_params = get_key_params(args)
    if args.setting is not None:
        setting_func = get_setting_func(args.setting)
        key_params = setting_func(key_params)

    params = ParamsBuilder(
            opt_use_fp16=False,
            **key_params).build_params()
    runner = EpochBasedRunner(**params)
    runner.train(args.train_upto_epoch)


if __name__ == '__main__':
    main()
