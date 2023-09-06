#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file ./config.yaml igc_main.py --train llm --num_train_epochs 1000 --use_accelerator \
	--llm latent --llm_log_level info --log_level info --device_map auto

#accelerate launch --config_file ./config.yaml igc_main.py --train llm --num_train_epochs 1000 --use_accelerator \
#	--llm latent --llm_log_level info --log_level info --device_map auto
#
#python -m torch.distributed.launch --help


accelerate launch --config_file ./configs/config.yaml trainer_state_encoder_only.py --train llm --num_train_epochs 1000 --use_accelerator \
	--llm latent --llm_log_level info --log_level info --device_map auto