import babylm_baseline_train.datasets.babyLM as babyLM
from babylm_baseline_train.configs.general import\
        add_func_in_general, get_general_data_func,\
        add_collate_fn_for_MLM
import functools
from itertools import product
import babylm_baseline_train.train.tk_funcs as tk_funcs


KWARGS = dict(
        all_things=globals(),
        specify_iter=[],
        specify_epoch=[5, 10, 20],
        )
DATA_KWARGS = dict(
        max_epochs=20, ckpt_save_interval=15,
        col_name='babyLM_10M')

def add_exp_seeds(
        exp_names, seeds, data_func,
        model_name='roberta-base',
        tokenizer=None,
        ):
    for exp_name, seed in zip(exp_names, seeds):
        if tokenizer is None:
            MLM_tokenizer = tk_funcs.get_roberta_tokenizer_func(
                    model_name=model_name)
        else:
            MLM_tokenizer = tokenizer
        add_func_in_general(
                func_name=exp_name,
                data_func=get_general_data_func(
                    data_func,
                    tokenizer=MLM_tokenizer,
                    **DATA_KWARGS),
                seed=seed,
                model_name=model_name,
                post_func=functools.partial(
                    add_collate_fn_for_MLM,
                    tokenizer=MLM_tokenizer),
                **KWARGS)

add_exp_seeds(
        exp_names=[
            'roberta_s1',
            ], 
        seeds=[1], 
        data_func=babyLM.get_babyLM_10M)

add_exp_seeds(
        exp_names=[
            'roberta_large_s1',
            ], 
        seeds=[1], 
        data_func=babyLM.get_babyLM_10M,
        model_name='roberta-large')
