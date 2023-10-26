import torch
import random, math
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error


class MLP(nn.Module):
    def __init__(self,in_dim, out_dim):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, out_dim)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = F.relu(self.fc1(x))
        pred = (self.fc2(x))

        if y is not None:
            loss = self.criterion(pred, y)
            return pred, loss
        else:
            return pred



class ActiveModel(nn.Module):
    def __init__(self, lstmdim=256, out_dim=1):
        super(ActiveModel, self).__init__()
        self.dr = MLP(1024, 256)
        self.rnn = nn.LSTMCell(258, lstmdim) 
        self.mlp = MLP(516, out_dim) 
    
    def forward(
        self, 
        train_X, train_y,
        SOTA, 
        former_X, former_y, 
        hx, cx
    ):
        newX = former_X[-1, :]  
        newy = former_y[-1, :].item()  
        shaped_SOTA = torch.FloatTensor([[SOTA]])
        shaped_newX = self.dr(newX.unsqueeze(dim=0))
        shaped_newy = torch.FloatTensor([[newy]])

        # historical experimental records and previously optima
        rnninput = torch.cat((shaped_SOTA, torch.cat((shaped_newX, shaped_newy), dim=1)), dim=1) 
        hx, cx = self.rnn(rnninput, (hx, cx)) 
        
        # diversity score
        train_X2 = (train_X ** 2).sum(dim=1).reshape((-1, 1))
        former_X2 = (former_X ** 2).sum(dim=1)
        D = train_X2 + former_X2 - 2 * train_X.mm(former_X.t())
        minD = torch.sqrt(torch.min(D, dim=1).values).reshape((-1, 1)) 

        # estimated properties and improvements
        max_pred = torch.max(train_y)  
        min_pred = torch.min(train_y)         
        prop = (train_y - min_pred) / (max_pred - min_pred + 1e-5) 

        inputs = torch.cat([hx.repeat(train_X.size(0), 1), 
                            prop, 
                            self.dr(train_X), train_y,  
                            minD, 
                            train_y - SOTA], 1)  
        pred = self.mlp(inputs).squeeze()
        score = nn.Softmax(dim=0)(pred)
        
        return score, hx, cx



def fitting_model(X, y, num_epoch=1000, early_stopping_rounds=5):
    """
    training XGBoostRegressor
    """
    rX = X.cpu().numpy()
    ry = y.cpu().numpy()
    # print("fitting model shape", rX.shape, ry.shape)
    reg = xgb.XGBRegressor(n_estimators=num_epoch, eval_metric=mean_squared_error)
    reg.fit(rX, ry, eval_set=[(rX, ry)], early_stopping_rounds=early_stopping_rounds, verbose=0)
    # print("xgb rmse: %.4f in %d" % (mean_squared_error(ry, reg.predict(rX), squared=False), reg.best_iteration))

    return reg, reg.best_iteration



def choose_experimental_x(size, scores, active_flag=True, epsilon=0.1, mode='train'):
    """Select a molecule to experiment """ 

    if not active_flag:
        return random.randint(0, size-1)
    if mode == "train":
        if random.random()>epsilon:
            return random.randint(0, size-1)
        else:
            return torch.argmax(scores).item()  
    elif mode == "test" or mode == "case" or mode == "custom":
        return torch.argmax(scores).item()  
    else:
        raise Exception("mode error!!!") 



def run_al_epoch(
    X, y, ids,
    GT_max_point, GT_min_point, 
    initial_point, 
    lstmdim,
    almodel, 
    hx, cx,
    mode,
    gamma=0.5,
    num_iter=10, 
    pri=True, 
    active_flag=True,
    epsilon=0.1,
):
    """An search of AlphaPharm"""
    search_space = X.shape[0]  
    X_dim = X.shape[1]  
    
    # Historical experimental records
    already_dataX = (initial_point.X).copy()  
    already_datay = (initial_point.y).copy()  
    already_dataid = np.array([initial_point.smi], dtype=object) 
    
    # Untested molecule candidates
    ready_dataX = X.copy()  
    ready_dataX = np.delete(ready_dataX, initial_point.idx, axis=0)  
    ready_datay = y.copy()  
    ready_datay = np.delete(ready_datay, initial_point.idx, axis=0)  
    ready_dataid = ids.copy()  
    ready_dataid = np.delete(ready_dataid, initial_point.idx, axis=0) 
    
    # Previously optimal record
    SOTA = initial_point.y.item()
    SOTA_mol = initial_point.smi

    logps = []
    rewards = []
    max_steps = 0  

    if pri:
        print("Begin identifying...")
        print("Step 0:")
        print("Random initialization: ", initial_point.smi)
        print("Property value: ", initial_point.y.item())



    if mode == 'train':
        maximum_trail_times = num_iter
    else:
        maximum_trail_times = search_space - 1 

    for i in range(maximum_trail_times):
        dist = math.fabs(SOTA - GT_max_point.y.item())/(GT_max_point.y.item() + 1e-5)
        if dist < 1e-5 and i >= num_iter:
            break

        datatensorX = torch.FloatTensor(already_dataX).detach()  
        datatensory = torch.FloatTensor(already_datay).detach() 

        # property predictor
        model, _ = fitting_model(datatensorX, datatensory, num_epoch=1000)
    
        train_X = torch.FloatTensor(ready_dataX)
        train_y = torch.FloatTensor(model.predict(ready_dataX)).unsqueeze(dim=1)

        # policy learning       
        scores, hx, cx = almodel(train_X, train_y, SOTA,
                                 datatensorX, datatensory,
                                 hx, cx)
        
        # select a molecule for experimentation 
        index = choose_experimental_x(train_X.size(0), scores, active_flag=active_flag, epsilon=epsilon, mode=mode)
        measuredX = train_X[index]  
        measuredy = ready_datay[index].item()  
        if pri:
            print("Step %d:"%(i+1))
            print("Selected molecule: ", ready_dataid[index])

        if len(scores.shape) == 0:   
            logp = torch.log(scores) # only left the last sample
            if pri:
                print("Property value: ", measuredy)
        else:
            logp = torch.log(scores[index])  
            if pri:
                print("Property value: ", measuredy)
        
        # compute the immediate reward
        if (measuredy > SOTA) and (math.fabs(measuredy-SOTA)>1e-5):
            reward = (measuredy - SOTA)/(GT_max_point.y.item() - initial_point.y.item() + 1e-5)
            SOTA = measuredy
            SOTA_mol = ready_dataid[index]

            dist = math.fabs(SOTA - GT_max_point.y.item())/(GT_max_point.y.item() + 1e-5)
            if dist < 1e-5:
                max_steps = i + 2  

        else:
            reward = 0


        # Update the historical experimental records and untested molecule candidates
        already_dataX = np.concatenate((already_dataX, ready_dataX[index].reshape(1, -1)))
        already_datay = np.concatenate((already_datay, ready_datay[index].reshape(1, -1)))
        already_dataid = np.concatenate((already_dataid, np.array([ready_dataid[index]], dtype=object)))
        ready_dataX = np.delete(ready_dataX, index, axis = 0)
        ready_datay = np.delete(ready_datay, index, axis = 0)
        ready_dataid = np.delete(ready_dataid, index, axis = 0)

        logps.append(logp)  
        rewards.append(reward)  
        
    # compute the returned reward
    retr = [0]*len(rewards)
    retr[-1] = rewards[-1]
    loss = 0
    
    for i in range(num_iter-1):
        id = num_iter-i-2
        retr[id] = gamma*retr[id+1] + rewards[id]
    
    for i in range(num_iter):
        loss = loss-retr[i]*logps[i]
    
    if pri:
        print("Information about the current candidate pool (100 molecules):")
        print("the identified molecule with the highest property value: ", SOTA_mol, SOTA)
        print("the ground truth molecule with the highest property value: ", GT_max_point.smi, GT_max_point.y.item())
    
    return loss, model, np.sum(rewards), SOTA, max_steps
