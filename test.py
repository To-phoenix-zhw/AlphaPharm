import torch, math, random
import numpy as np
import pandas as pd

from data import *
from models import *


def test(
    datasets_list,
    dataset_num,
    save_path,
    test_path,
    mode,
    lstmdim=256,
    search_space=1000,
    searchtimes=1, 
    num_iter=10,
    optnum=20, 
    maxepoch=1,
    gamma=0.5,
    pri=True,
    active_flag=True,
    epsilon=0.1,
):
    almodel = torch.load(save_path + test_path)
    hx = torch.randn(1, lstmdim) 
    cx = torch.randn(1, lstmdim) 

    allosses = 0
    alres = 0
    alsotas = 0
    aldises = 0

    # choose the task
    task_obj = 0
    dataset_obj_idx = dataset_num
    y_obj_no = 0
    dataset = datasets_list[dataset_obj_idx]

    # random sample 100 molecules
    random_choose_X_index = np.random.randint(0, len(dataset), search_space)
    dataset = dataset.select(random_choose_X_index)
    X = dataset.X
    ids = dataset.ids
    if len(dataset.y.shape) == 1:
        y = dataset.y
        task_name = dataset.get_task_names()[0]
    else:
        if dataset.y.shape[1] != 1:
            y = dataset.y[:,y_obj_no]
            task_name = dataset.get_task_names()[y_obj_no]
        else:
            y = dataset.y
            task_name = dataset.get_task_names()[0]
    y = y.reshape(-1, 1)


    if X.shape[0] != search_space:
        raise Exception("X error!!!") 
    
    print("*****TASK %d %s *****" % (task_obj, task_name)) 
    print(X.shape, y.shape, ids.shape)  # X; [100, 1024]  y: [100, 1] id: [100,]

    # Maxima, minima, and starting molecule in the current search space
    GT_max_point, GT_min_point, initial_point = compute_extreme_and_initial_point(X, y, ids, search_space)

    print("-"*5 + "Reinforcement-Active Learning" + "-"*5)
    alloss, alxgbmodel, alre, alsota = run_al_epoch(
        X, y, ids,
        GT_max_point, GT_min_point, 
        initial_point, 
        lstmdim,
        almodel, 
        hx, cx,
        mode,
        gamma,
        num_iter,
        pri, 
        active_flag=True,
        epsilon=0.1,
    )
    aldis = math.fabs(alsota - GT_max_point.y.item())/(GT_max_point.y.item() + 1e-5)

    allosses += alloss
    alres += alre
    alsotas += alsota
    aldises += aldis
    print("this search reward %.4f" % alre)
    print("---Ending the search(%d experiments)---" % (num_iter))


    return alres, alsotas, aldises
