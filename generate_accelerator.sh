echo '{
"compute_environment": "LOCAL_MACHINE",
"distributed_type": "MULTI_GPU",
"downcast_bf16": "no",
"gpu_ids": "all",
"machine_rank": 0,
"main_training_function": "main",
"mixed_precision": "no",
"num_machines": 1,
"num_processes": 2,
"rdzv_backend": "static",
"same_network": true,
"tpu_env": [],
"tpu_use_cluster": false,
"tpu_use_sudo": false,
"use_cpu": false
}' > config.yaml
