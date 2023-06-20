#!/bin/bash

accelerate launch --config_file ./config.yaml igc_main.py --train llm --num_train_epochs 1000 \
	--llm latent --llm_log_level info --log_level info
