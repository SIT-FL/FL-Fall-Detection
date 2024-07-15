#!/bin/sh

python3 main.py \
    --exp_name FL_FD --seed 123456 --device cpu \
    --dataset FEMNIST \
    --split_type pre --test_size 0.2 \
    --model_name FEMNISTCNN --resize 28 --hidden_size 64 \
    --algorithm fedavg --eval_fraction 1 --eval_type both --eval_every 50 --eval_metrics acc1 acc5 f1 \
    --R 10 --E 10 --C 0.003 --B 64 --beta1 0 \
    --optimizer SGD --lr 0.001 --lr_decay 1 --lr_decay_step 1 --criterion CrossEntropyLoss

python3 main.py \
    --exp_name FedAvg_LEAF_FEMNIST --seed 42 --device cpu \
    --dataset FEMNIST \
    --split_type pre --test_size 0.1 \
    --model_name FEMNISTCNN --resize 28 --hidden_size 64 \
    --algorithm fedavg --eval_fraction 1 --eval_type both --eval_every 10 --eval_metrics acc1 acc5 precision recall f1 \
    --R 50 --E 5 --C 0.003 --B 10 --beta1 0 \
    --optimizer SGD --lr 0.0003 --lr_decay 1 --lr_decay_step 1 --criterion CrossEntropyLoss