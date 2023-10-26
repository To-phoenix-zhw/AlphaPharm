#! /bin/bash
python -u run.py --mode train --save_path ./results/run_train
# CUDA_VISIBLE_DEVICES=5 python -u run.py --mode train --device cuda:0 --searchtimes 1 --save_path ./results/run_train
