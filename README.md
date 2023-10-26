# <a name="AlphaPharm"></a> AlphaPharm: a reinforcement-active learning framework to streamline lead optimization in drug discovery

Official PyTorch implementation of paper "AlphaPharm: a reinforcement-active learning framework to streamline lead optimization in drug discovery". 

[AlphaPharm](#AlphaPharm)

- [Overview](#overview)
- [The testing of AlphaPharm'](#obutton)
- [The training of AlphaPharm](#otraining)
- [More information](#more)
  - [Installation](#installation)
  - [Datasets](#datasets)
  - [Training](#training)
  - [Testing](#testing)
  - [Contact](#contact)


## <a name="overview"></a>Overview

AlphaPharm is a novel reinforcement-active learning framework that systematically and adaptively identifies promising drug candidates. Mimicking the cyclical cognitive process of decision-making via trial and error, AlphaPharm integrates the stages of molecular hypothesis, wet-lab testing, few-shot learning, and policy refinement.

Specifically, AlphaPharm contains two main modules, the property predictor and the policy network. At each decision-making iteration, the property predictor takes the fingerprints of the candidate molecules and estimates their property values. The policy network accepts the molecules, their estimated property values, and previous optimal records as inputs to decide which candidate should be tested next. 

Thanks to the above design, AlphaPharm significantly outperforms all the current active learning and few-shot learning methods. It should be noted that AlphaPharm astoundingly singled out a drug candidate, AA-35, with the most potent analgesic activity, from a pool of 51 compounds in just six trials. These findings validate proof-of-concept and highlight the promise of AlphaPharm as a powerful AI-driven computational tool for drug discovery.
<img src="./figure/Figure1.svg">







## <a name="obutton"></a>The testing of AlphaPharm
First, you need to simply accomplish two steps to loading the data.

1. Download the dataset archive `data.zip` from [this link](https://drive.google.com/drive/folders/1mPZCfQl5gKSgLEwnwMkyjgDidJaTbXgg?usp=share_link).
2. Extract the ZIP archive using the command: `unzip data.zip`.



Then, you can test the AlphaPharm model. 

As soon as you execute `bash run.sh`, the testing process will be started, performing the molecule identification process for certain properties with the trained AlphaPharm model. You will get the performance at the bottom of a log file with the following formats: 

>********Statistic Performance********
>
>Average success rate: [a percentage]
>
>Average search steps: [an integer]
>
>time cost [a floating point number] s

If you want to test AlphaPharm on the other properties, you can edit the `test.sh` file by revising the value of the `--task_id` argument. (Domain of this argument: [0, 1, 2])



## <a name="otraining"></a>The training of AlphaPharm

If you want to train your own AlphaPharm from scratch, just change `test` to `train` in the `run.sh` file. Then you will see the training process with the following formats:

>[2023-10-24 15:30:52,632::train::INFO] Building model...
>
>[2023-10-24 15:30:52,640::train::INFO] Training model...
>
>[2023-10-24 15:31:52,644::train::INFO] [Train] Iter 1 | reward [a floating point number]
>
>[2023-10-24 15:32:53,329::train::INFO] [Train] Iter 2 | reward [a floating point number]
>
>[2023-10-24 15:32:53,329::train::INFO] [Train] Iter 3 | reward [a floating point number]




## <a name="more"></a>More information about AlphaPharm

### <a name="installation"></a>Installation

#### Dependency

The code has been implemented in the following environment:

| Package        | Version  |
| -------------- | -------- |
| Python         | 3.7      |
| PyTorch        | 1.8.0    |
| CUDA           | 11.1     |
| RDKit          | 2022.9.1 |
| DeepChem       | 2.7.1    |
| py-xgboost-gpu | 1.5.1    |

The code should work with Python >= 3.7. You could change the package version according to your need.



#### Install via Conda and Pip

```shell
conda create -n AlphaPharm python=3.7
conda activate AlphaPharm

pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
conda install cudatoolkit=11.1 cudnn

pip install rdkit
pip install deepchem
pip install openpyxl
conda install -c conda-forge py-xgboost-gpu
```



### <a name="datasets"></a>Datasets

To train and analyze the AlphaPharm model, we constructed a large-scale dataset from MoleculeNet and ChEMBL. Due to the storage limitation in Github, the complete data are organized in the [data](https://drive.google.com/drive/folders/1mPZCfQl5gKSgLEwnwMkyjgDidJaTbXgg?usp=share_link) Google Drive folder. The raw data is in the `raw_data` folder and the data directly loaded by the model is stored in `data.zip`.  




### <a name="training"></a>Training

#### Training from scratch

Researchers could train their own AlphaPharm from scratch with the following bash order.

```bash
python run.py --mode train --save_path [saved_model_path]
```

For example:

```bash
python -u run.py --mode train --save_path results/run_train
```



#### Trained model checkpoint

We uploaded the trained model to the `checkpoints` folder.



### <a name="testing"></a>Testing

#### Testing on the dataset

Researchers could test the model on the test dataset.

```bash
python -u run.py --mode test --searchtimes 1 --test_times [times of testing] --save_path [saved_model_path] --test_path [saved_model_name] --pri true --task_id [task_id]
```

For example:

```bash
python -u run.py --mode test --searchtimes 1 --test_times 100 --save_path checkpoints --test_path almodel_75000.pt --pri true --task_id 0
```



### <a name="contact"></a>Contact

If you encounter any problems during the setup of environment or the execution of AlphaPharm, do not hesitate to contact [liuxianggen@scu.edu.cn](mailto:liuxianggen@scu.edu.cn) or [hanwenzhang@stu.scu.edu.cn](mailto:hanwenzhang@stu.scu.edu.cn). You could also create an issue under the repository: https://github.com/To-phoenix-zhw/AlphaPharm.
