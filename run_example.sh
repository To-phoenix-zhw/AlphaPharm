#! /bin/bash
python -u run.py --searchtimes 1 --mode case  --checkpoint_path checkpoints/almodel_75000.pt  --pri true --task_id 0 --test_path ./data/dataset/testing-set  --num_iter 40
