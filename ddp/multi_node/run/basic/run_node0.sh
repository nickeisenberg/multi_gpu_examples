torchrun \
    --nproc_per_node=gpu \
    --nnodes=2 \
    --node_rank=0 \
    --rdzv_id=456 \
    --rdzc_backend=c10d \
    --rdzv_endpoint=<host>:<port> \
    ../../ddp_multi_node_torchrun.py
