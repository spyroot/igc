#!/bin/bash
python trainer.py --train llm --num_train_epochs 10 --llm latent --llm_log_level info --log_level info

python trainer.py --train llm --num_train_epochs 10 --llm encoder --llm_log_level info --log_level info


CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file /home/spyroot/.cache/huggingface/accelerate/default_config.yaml trainer.py

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file /home/spyroot/.cache/huggingface/accelerate/default_config.yaml trainer.py \
--train llm \
--num_train_epochs 10 \
--llm encoder \
--llm_log_level info \
--log_level info