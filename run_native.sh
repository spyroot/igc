#!/bin/bash

deepspeed --num_gpus=2 your_program.py native.py --deepspeed ds_config_zero3.json


#python -c 'from transformers import AutoModel; \
#from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live; \
#model = AutoModel.from_pretrained("gpt2-xl"); \
#estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=2, num_nodes=1)'

#from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live; \
#model = AutoModel.from_pretrained("gpt2-xl"); \
#estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=2, num_nodes=1)'
#Estimated memory needed for params, optim states and gradients for a:
#HW: Setup with 1 node, 2 GPUs per node.
#SW: Model with 1557M total params, 80M largest layer params.
#  per CPU  |  per GPU |   Options
#   39.17GB |   0.30GB | offload_param=cpu , offload_optimizer=cpu , zero_init=1
#   39.17GB |   0.30GB | offload_param=cpu , offload_optimizer=cpu , zero_init=0
#   34.82GB |   1.75GB | offload_param=none, offload_optimizer=cpu , zero_init=1
#   34.82GB |   1.75GB | offload_param=none, offload_optimizer=cpu , zero_init=0
#    0.90GB |  13.36GB | offload_param=none, offload_optimizer=none, zero_init=1
#   17.41GB |  13.36GB | offload_param=none, offload_optimizer=none, zero_init=0

export BS=16; \
rm -rf output_dir; \
deepspeed --num_gpus=1 run_translation2.py \
--output_dir output_dir \
--adam_eps 1e-06 \
--evaluation_strategy=steps \
--label_smoothing 0.1 \
--learning_rate 3e-5 \
--logging_first_step \
--logging_steps 1000 \
--num_train_epochs 1 \
--overwrite_output_dir  \
--per_device_train_batch_size $BS \
--per_device_eval_batch_size $BS \
--predict_with_generate \
--sortish_sampler \
--val_max_target_length 128 \
--warmup_steps 500 \
--max_train_samples 2000 \
--max_eval_samples 500 \
--deepspeed ds_config_zero3.json --fp16


#export BS=20; rm -r output_dir; CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../src USE_TF=0 deepspeed --num_gpus=1 \
#./finetune_trainer.py \
# --model_name_or_path \
# t5-3b --output_dir output_dir --adam_eps 1e-06 --data_dir wmt_en_ro \
#--do_eval --do_predict --do_train --evaluation_strategy=steps --freeze_embeds --label_smoothing 0.1 --learning_rate 3e-5 \
#--logging_first_step --logging_steps 1000 --max_source_length 128 --max_target_length 128 --num_train_epochs 1 \
#--overwrite_output_dir --per_device_eval_batch_size $BS --per_device_train_batch_size $BS --predict_with_generate \
#--eval_steps 25000  --sortish_sampler --task translation_en_to_ro --test_max_target_length 128 \
#--val_max_target_length 128 --warmup_steps 5 --n_train 60 --n_val 10 --n_test 10 --deepspeed ds_config_1gpu.json --fp16
