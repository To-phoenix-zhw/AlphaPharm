import random
import numpy as np
import pandas as pd
import deepchem as dc


class Point(object):
    def __init__(self, idx, X, y, smi):
        self.idx = idx
        self.X = X
        self.y = y
        self.smi = smi
    def __str__(self):
        return "{}, {}".format(self.y, self.smi)




def load_train_molnet_datasets():
    train_molnet_datasets = []
    molnet_root = "./datasets/molnet/normECFP/"
    train_tasks_df = pd.read_excel(molnet_root + 'train_tasks.xlsx')
    train_tasks = train_tasks_df.values.squeeze().tolist()
    
    for part_concat in train_tasks:
        train_molnet_datasets.append(dc.data.DiskDataset(molnet_root + part_concat))
    return train_molnet_datasets


def load_test_molnet_datasets():
    test_molnet_datasets = []
    molnet_root = "./datasets/molnet/normECFP/"
    test_tasks_df = pd.read_excel(molnet_root + 'test_tasks.xlsx')
    test_tasks = test_tasks_df.values.squeeze().tolist()
    
    for part_concat in test_tasks:
        test_molnet_datasets.append(dc.data.DiskDataset(molnet_root + part_concat))
    return test_molnet_datasets


def load_train_chembl_datasets():
    train_chembl_datasets = []
    chembl_root = "./datasets/ChEMBL/"
    pre_part = "./datasets/ChEMBL/norm_pValue/"
    train_address = pd.read_csv(chembl_root + 'train_address.csv').squeeze().values.tolist()
    
    for part_concat in train_address:
        train_chembl_datasets.append(dc.data.DiskDataset(pre_part + part_concat))
    return train_chembl_datasets


def load_test_chembl_datasets():
    test_chembl_datasets = []
    chembl_root = "./datasets/ChEMBL/"
    pre_part = "./datasets/ChEMBL/norm_pValue/"
    test_address = pd.read_csv(chembl_root + 'test_address.csv').squeeze().values.tolist()
    
    for part_concat in test_address:
        test_chembl_datasets.append(dc.data.DiskDataset(pre_part + part_concat))
    return test_chembl_datasets



def load_train_datasets():
    train_molnet_datasets = load_train_molnet_datasets()
    train_chembl_datasets = load_train_chembl_datasets()
    train_datasets_list = train_molnet_datasets + train_chembl_datasets
    return train_datasets_list



def load_test_datasets():
    test_molnet_datasets = load_test_molnet_datasets()
    test_chembl_datasets = load_test_chembl_datasets()
    test_datasets_list = test_molnet_datasets + test_chembl_datasets
    return test_datasets_list




def compute_extreme_and_initial_point(X, y, ids, search_space=100):
    """Calculate the points corresponding to the maximum and minimum true values. 
    Random initialize a starting point."""
    
    X_dim = X.shape[1] 
    maxy_index = np.argmax(y)
    maxy_X = X[maxy_index].reshape(-1, X_dim)  
    maxy_y = y[maxy_index].reshape(-1, 1) 
    maxy_id = ids[maxy_index]  
    GT_max_point = Point(maxy_index, maxy_X, maxy_y, maxy_id)
    print("GT max Point: ", GT_max_point.smi)
    print(GT_max_point.y.item())

    miny_index = np.argmin(y)
    miny_X = X[miny_index].reshape(-1, X_dim)
    miny_y = y[miny_index].reshape(-1, 1)
    miny_id = ids[miny_index]
    GT_min_point = Point(miny_index, miny_X, miny_y, miny_id)
    print("GT min Point: ", GT_min_point.smi)
    print(GT_min_point.y.item())


    randomindex = random.randint(0, search_space-1)
    initial_X = X[randomindex].reshape(-1, X_dim)  # [1, 1024]
    initial_y = y[randomindex].reshape(-1, 1)  # [1, 1]
    initial_id = ids[randomindex]  # a string
    initial_point = Point(randomindex, initial_X, initial_y, initial_id)
    print("Random Chosen Initial Point", initial_point.smi)
    print(initial_point.y.item())
    
    return GT_max_point, GT_min_point, initial_point 



def compute_extreme_and_initial_point_appoint(X, y, ids, initial=0):
    """Calculate the points corresponding to the maximum and minimum true values. 
    Specify a molecule as the initial point."""
    X_dim = X.shape[1]  
    maxy_index = np.argmax(y)
    maxy_X = X[maxy_index].reshape(-1, X_dim)  # [1, 1024]
    maxy_y = y[maxy_index].reshape(-1, 1)  # [1, 1]
    maxy_id = ids[maxy_index]  # a string
    GT_max_point = Point(maxy_index, maxy_X, maxy_y, maxy_id)
    print("GT max Point: ", GT_max_point.smi)
    print(GT_max_point.y.item())

    miny_index = np.argmin(y)
    miny_X = X[miny_index].reshape(-1, X_dim)
    miny_y = y[miny_index].reshape(-1, 1)
    miny_id = ids[miny_index]
    GT_min_point = Point(miny_index, miny_X, miny_y, miny_id)
    print("GT min Point: ", GT_min_point.smi)
    print(GT_min_point.y.item())

    randomindex = initial
    initial_X = X[randomindex].reshape(-1, X_dim)  # [1, 1024]
    initial_y = y[randomindex].reshape(-1, 1)  # [1, 1]
    initial_id = ids[randomindex]  # a string
    initial_point = Point(randomindex, initial_X, initial_y, initial_id)
    print("Chosen Initial Point", initial_point.smi)
    print(initial_point.y.item())
    
    return GT_max_point, GT_min_point, initial_point 