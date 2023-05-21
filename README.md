# igc
Infrastructure Goal Condition  Reinforce Learner

```bash
conda create -n igc python=3.10
conda activate igc
conda install pytorch::pytorch torchvision torchaudio -c pytorch
pip install 'transformers[torch]'
pip install deepspeed
pip install fairscale
pip install asv
pip install pynvml

```

```bash
CFLAGS=-noswitcherror pip install mpi4py
CC=cc CXX=CC pip install mpi4py -U          
```


MODEL_NAME=gpt2-xl
PER_DEVICE_TRAIN_BATCH_SIZE=1
HF_PATH=~/projects
NEPOCHS=1
NGPUS=2
NNODES=1
MAX_STEPS=50
OUTPUT_DIR=./output_b${PER_DEVICE_TRAIN_BATCH_SIZE}_g${NGPUS}_$MAX_STEPS


deepspeed --num_gpus=2 run_clm.py \
--deepspeed ../dsconfigs/ds_config_fp16_z2.json\
--model_name_or_path $MODEL_NAME \
--dataset_name wikitext \
--dataset_config_name wikitext-2-raw-v1 \
--do_train \
--fp16 \
--per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
--learning_rate 2e-5 \
--num_train_epochs $NEPOCHS \
--output_dir ${OUTPUT_DIR}_z2 \
--overwrite_output_dir \
--save_steps 0 \
--max_steps $MAX_STEPS \
--save_strategy "no"