import babylm_baseline_train.datasets.babyLM as babyLM
from babylm_baseline_train.configs.general import\
        add_func_in_general_for_opt, get_general_data_func
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
        model_name=None,
        tokenizer=None,
        ):
    for exp_name, seed in zip(exp_names, seeds):
        add_func_in_general_for_opt(
                func_name=exp_name,
                data_func=get_general_data_func(
                    data_func,
                    tokenizer=tokenizer,
                    **DATA_KWARGS),
                seed=seed,
                model_name=model_name,
                **KWARGS)

add_exp_seeds(
        exp_names=[
            'opt125m_s1',
            'opt125m_s2',
            'opt125m_s3',
            'opt125m_s4',
            ], 
        seeds=[1, 2, 3, 4], 
        data_func=babyLM.get_babyLM_10M)

add_exp_seeds(
        exp_names=[
            'opt350m_s1',
            'opt350m_s2',
            'opt350m_s3',
            'opt350m_s4',
            ], 
        seeds=[1, 2, 3, 4], 
        data_func=babyLM.get_babyLM_10M,
        model_name='350m')
