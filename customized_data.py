import os
import numpy as np
import pandas as pd
from rdkit import Chem
import deepchem as dc

data = pd.read_excel('./molecules.xlsx')

data_path = "./customized_data"
smiles = data["Smiles"].values
pro = data["Values"].values
featurizer = dc.feat.CircularFingerprint(size=1024)  # 'ECFP'
features = featurizer.featurize(smiles)
created_dataset = dc.data.NumpyDataset(X=features, y=pro, ids=smiles)
print(created_dataset)
created_disk_dataset = dc.data.DiskDataset.from_numpy(X=features, y=pro, ids=smiles, tasks=["customized_research"], data_dir=data_path)
print(created_disk_dataset)