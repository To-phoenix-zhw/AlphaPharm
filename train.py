import torch, math, random
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from models import *
from data import *
from utils import *

def train(
    datasets_list,
    save_path,
    model_path,
    continue_epoch,
    continue_rewards,
    continue_distances,
    mode,
    logger,
    device,
    lstmdim=256,
    search_space=50,
    searchtimes=3, 
    num_iter=10,
    optnum=20, 
    maxepoch=20000,
    gamma=0.5,
    pri=False,
    active_flag=True,
    epsilon=0.1,
):
    rewards = 0  
    distances = 0  
    rewards_list = [] 
    models_list = [] 
    SOTAs_list = []  

    logger.info('Building model...')
    checkpoint_path = get_checkpoint_dir(save_path)

    almodel = ActiveModel(lstmdim).to(device)
    if model_path != '':
        almodel = torch.load(model_path)
        # print("Load Model: %s, continue training." % (model_path))
        rewards = continue_rewards*continue_epoch
        distances = continue_distances*continue_epoch
    
    aloptimizer = optim.Adam(almodel.parameters())  
    almodel.train() 
    hx = torch.randn(1, lstmdim) 
    cx = torch.randn(1, lstmdim) 
    
    train_nos = list(range(len(datasets_list)))
    train_loop = tqdm(range(maxepoch), desc='Training')

    logger.info('Training model...')
    for cur_epo in train_loop:
        # random choose a task
        dataset_obj_idx = np.random.choice(train_nos)
        dataset = datasets_list[dataset_obj_idx]

        # random sample 100 molecules
        random_choose_X_index = np.random.randint(0, len(dataset), search_space)
        dataset = dataset.select(random_choose_X_index)
        X = dataset.X
        ids = dataset.ids
        y = dataset.y.reshape(-1, 1)
        task_name = dataset.get_task_names()[0]

        if X.shape[0] != search_space:
            raise Exception("X error!!!") 
            
        # print("*****EPOCH %d TASK %d %s *****" % (cur_epo+1, dataset_obj_idx, task_name)) 
        # print(X.shape, y.shape, ids.shape)  

        if cur_epo < continue_epoch:
            continue

        reward_list = []  
        loss_list = [] 
        model_list = [] 
        SOTA_list = [] 
        
        # Maxima, minima, and starting molecule in the current search space
        GT_max_point, GT_min_point, initial_point = compute_extreme_and_initial_point(X, y, ids, search_space)
    
        # search
        for j in range(searchtimes):
            # print("---Into the search " + str(j+1) +  " (%d experiments)---" % num_iter)
            loss, model, re, sota, step = run_al_epoch(
                X, y, ids,
                GT_max_point, GT_min_point, 
                initial_point, 
                lstmdim,
                almodel, 
                hx, cx,
                mode,
                device,
                gamma,
                num_iter,
                pri, 
                active_flag,
                epsilon,
            )
            loss_list.append(loss)
            reward_list.append(re)
            model_list.append(model)
            SOTA_list.append(sota)
            # print("this search reward %.4f" % re)
            # print("---Ending the search(10 experiments)---")
        
        # Gradients: only use losses of search processes whose reward is bigger than the average 
        cur_re = 0
        cnt_re = 0
        cur_dis = 0
        for id, re in enumerate(reward_list):
            if (re > np.mean(reward_list)) or (math.fabs(re-np.mean(reward_list))<1e-5):  
                (loss_list[id] / optnum).backward()  
                cur_re += re
                cnt_re += 1
                cur_dis += (math.fabs(SOTA_list[id] - GT_max_point.y.item())/(GT_max_point.y.item() + 1e-5))

        
        # Update parameters every 20 episodes
        if cur_epo % optnum == optnum - 1: 
            clip_grad_norm_(almodel.parameters(), 2.0)  
            aloptimizer.step()  
            aloptimizer.zero_grad() 
            almodel.zero_grad()  
        
        rewards += (cur_re/cnt_re)
        distances += (cur_dis/cnt_re)
        rewards_list.append(rewards/(cur_epo+1))
        models_list.append(model_list[np.argmax(reward_list)])
        SOTAs_list.append(SOTA_list[np.argmax(SOTA_list)])

        logger.info('[Train] Iter %d | reward %.6f' % (cur_epo+1, rewards/(cur_epo+1)))
        if ((cur_epo+1)==1) or ((cur_epo+1)%optnum==0):
             # Save models every 20 episodes         
            torch.save(almodel, checkpoint_path + '/almodel_'+ str(cur_epo+1) + '.pt')
        
        train_loop.set_description(f'Iter [{cur_epo+1}/{maxepoch}]')
        train_loop.set_postfix(reward = rewards/(cur_epo+1))
        # print("*****ENDING THE EPOCH %d TASK %d %s *****" % (cur_epo+1, dataset_obj_idx, task_name)) 

    return rewards_list
