python /home/octavians/DeePHFProject/train.py \
    --config-path /home/octavians/DeePHFProject/gnn_config.yml \
    --exp-name qm7b_T_v3 \
    --seed 42 \
    --batch-size 256 \
    --train-data-path /home/octavians/DeePHFProject/hf_mobml_datasets/split_qm7b_T/train \
    --val-data-path /home/octavians/DeePHFProject/hf_mobml_datasets/split_qm7b_T/val \
    --num-workers 4