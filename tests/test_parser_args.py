from igc.shared.shared_main import shared_main


def main():
    args = shared_main()
    print(args)


if __name__ == '__main__':
    main()
# Namespace(per_device_train_batch_size=8,
#           per_device_eval_batch_size=8,
#           auto_batch=False,
#           num_train_epochs=10,
#           max_train_steps=None,
#           output_dir='experiments/gpt2-medium_8_AdamW_StepLR_lr_5e-05/2023-06-02_05-29-09',
#           seed=42,
#           data_seed=42,
#           train=True,
#           eval=True,
#           predict=False,
#           max_grad_norm=1.0,
#           max_steps=-1,
#           optimizer='AdamW',
#           learning_rate=5e-05,
#           weight_decay=0.0,
#           gradient_checkpointing=True,
#           gradient_accumulation_steps=8,
#           scheduler='StepLR',
#           num_warmup_steps=0,
#           log_dir='experiments/gpt2-medium_8_AdamW_StepLR_lr_5e-05/2023-06-02_05-29-09/../../logs',
#           llm_log_level='warning',
#           log_strategy='epoch',
#           log_on_each_node=False,
#           log_steps=10,
#           save_strategy='epoch',
#           save_steps=500,
#           save_total_limit=None,
#           save_on_each_node=False,
#           bf16=False, fp16=True,
#           fp16_opt_level='O1',
#           half_precision_backend='auto',
#           bf16_full_eval=False,
#           fp16_full_eval=False,
#           tf32=False,
#           ddp_backend='nccl',
#           sharded_ddp='simple',
#           deepspeed=None,
#           dataloader_pin_memory=False,
#           num_workers=0,
#           report_to='tensorboard',
#           model_type='gpt2-medium',
#           local_rank=-1,
#           device=device(type='cuda', index=0))
