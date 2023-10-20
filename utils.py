import os, random
import torch
import numpy as np
import pandas as pd


def mksure_path(dirs_or_files):
    if not os.path.exists(dirs_or_files):
        os.makedirs(dirs_or_files)



def set_seed(seed=0):
    random.seed(seed)  # python
    np.random.seed(seed)  # numpy
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # GPU
    print("set seed %d" % seed)