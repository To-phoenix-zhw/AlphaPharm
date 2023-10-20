import torch, math, random
import numpy as np
import pandas as pd

from data import *
from models import *


def case(
    dataset,
    save_path,
    test_path,
    mode,
    initial=0,
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

    X = dataset.X
    ids = dataset.ids
    y = dataset.y.reshape(-1, 1)
    task_name = dataset.get_task_names()[0]

    print("*****TASK %s *****" % (task_name)) 
    print(X.shape, y.shape, ids.shape)  

    # Maxima, minima, and starting molecule in the current search space
    GT_max_point, GT_min_point, initial_point = compute_extreme_and_initial_point_appoint(X, y, ids, initial)


    print("-"*5 + "Reinforcement-active Learning" + "-"*5)
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
