#!/bin/bash

for size in {10M,100M}; do
python run_t5_mlm_flax.py \
	--output_dir ../babyLM_${size}/t5_s1/ \
	--model_type t5 \
	--config_name ../src/babylm_baseline_train/configs/BabyLM/exp_strict_encdec/ \
	--tokenizer_name t5-base \
	--train_file $BABYLM_ROOT_DIR/babylm_${size}/babylm_${size}.txt \
	--validation_file $BABYLM_ROOT_DIR/babylm_dev/babylm_dev.txt \
	--max_seq_length 128 \
	--per_device_train_batch_size='128' \
	--per_device_eval_batch_size='16' \
	--adafactor \
	--learning_rate='0.005' \
	--weight_decay='0.001' \
	--warmup_steps='2000' \
	--overwrite_output_dir \
	--logging_steps='2500' \
	--save_epochs='2' \
	--eval_epochs='2' \
	--num_train_epochs='20' \
	--seed 1
done
