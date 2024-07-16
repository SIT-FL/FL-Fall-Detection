#!/bin/sh

# FEMNIST (only supports local evaluation)
# FedAvg
python3 main.py \
    --exp_name FL_FD --seed 123456 --device cpu \
    --dataset FEMNIST \
    --split_type pre --test_size 0.2 \
    --model_name FEMNISTCNN --resize 28 --hidden_size 64 \
    --algorithm fedavg --eval_fraction 1 --eval_type local --eval_every 10 --eval_metrics acc1 acc5 precision recall f1 \
    --R 10 --E 10 --C 0.003 --B 64 --beta1 0 \
    --optimizer SGD --lr 0.001 --lr_decay 1 --lr_decay_step 1 --criterion CrossEntropyLoss

# FedProx
python3 main.py \
    --exp_name FL_FD_Prox --seed 123456 --device cpu \
    --dataset FEMNIST \
    --split_type pre --test_size 0.2 \
    --model_name FEMNISTCNN --resize 28 --hidden_size 64 \
    --algorithm fedprox --eval_fraction 1 --eval_type local --eval_every 10 --eval_metrics acc1 acc5 precision recall f1 \
    --R 10 --E 10 --C 0.003 --B 64 --beta1 0 \
    --optimizer SGD --lr 0.001 --lr_decay 1 --lr_decay_step 1 --criterion CrossEntropyLoss

# LSTM
python3 main.py \
    --exp_name Sent140_FedAvg_Fixed --seed 42 --device cuda:2 \
    --dataset Sent140 --learner fixed \
    --split_type pre --rawsmpl 0.01 --test_size 0.2 \
    --model_name Sent140LSTM --embedding_size 300 --hidden_size 80 --num_layers 2 \
    --algorithm fedavg --eval_fraction 1 --eval_type local --eval_every 1000 --eval_metrics acc1 \
    --R 1000 --C 0.0021 --E 5 --B 10 \
    --optimizer SGD --lr 0.0003 --lr_decay 0.9995 --lr_decay_step 1 --criterion BCEWithLogitsLoss