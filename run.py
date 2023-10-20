import pickle, time
import numpy as np
from optparse import OptionParser

from utils import *
from data import *
from train import *
from test import *
from case import *


parser = OptionParser()
parser.add_option("--seed", dest="seed", default=0)
parser.add_option("--lstmdim", dest="lstmdim", default=256)
parser.add_option("--search_space", dest="search_space", default=100)
parser.add_option("--searchtimes", dest="searchtimes", default=3)
parser.add_option("--num_iter", dest="num_iter", default=10)
parser.add_option("--optnum", dest="optnum", default=20)
parser.add_option("--maxepoch", dest="maxepoch", default=160000)
parser.add_option("--gamma", dest="gamma", default=0.5)
parser.add_option("--pri", dest="pri", default='true')
parser.add_option("--active_flag", dest="active_flag", default='true')
parser.add_option("--epsilon", dest="epsilon", default=0.1)
parser.add_option("--mode", dest="mode", default='train')
parser.add_option("--custom", dest="custom", default='')
parser.add_option("--case_task", dest="case_task", default=0)
parser.add_option("--data_path", dest="data_path", default='datasets')
parser.add_option("--save_path", dest="save_path", default='')
parser.add_option("--model_path", dest="model_path", default='')
parser.add_option("--continue_epoch", dest="continue_epoch", default=0)
parser.add_option("--continue_rewards", dest="continue_rewards", default=0)
parser.add_option("--continue_distances", dest="continue_distances", default=0)
parser.add_option("--test_path", dest="test_path", default='')
parser.add_option("--test_times", dest="test_times", default=100)
parser.add_option("--begin", dest="begin", default=0)
parser.add_option("--end", dest="end", default=0)
opts,args = parser.parse_args()
print(opts)



seed = int(opts.seed) 
lstmdim = int(opts.lstmdim)  
search_space = int(opts.search_space)  
searchtimes = int(opts.searchtimes)  
num_iter = int(opts.num_iter)  
optnum = int(opts.optnum) 
maxepoch = int(opts.maxepoch) 
gamma = float(opts.gamma) 
pri = True if (opts.pri == 'true') else False 
active_flag = True if (opts.active_flag == 'true') else False
epsilon = float(opts.epsilon)  
mode = str(opts.mode)
custom = str(opts.custom)  
case_task = int(opts.case_task)  
data_path = str(opts.data_path) + "/"
save_path = str(opts.save_path) + "/"
model_path = str(opts.model_path) 
continue_epoch = int(opts.continue_epoch)
continue_rewards = float(opts.continue_rewards)
continue_distances = float(opts.continue_distances)
test_path = str(opts.test_path)  
test_times = int(opts.test_times) 
begin = int(opts.begin) 
end = int(opts.end)

mksure_path(save_path)
mksure_path(data_path)



if __name__ == "__main__":
    time_start = time.time()
    set_seed(seed)

    if mode == "train":
        datasets_list = load_train_datasets()
        print(len(datasets_list))
        rewards_list = train(
            datasets_list,
            save_path,
            model_path,
            continue_epoch,
            continue_rewards,
            continue_distances,
            mode,
            lstmdim,
            search_space,
            searchtimes, 
            num_iter,
            optnum,
            maxepoch,
            gamma,
            pri,
            active_flag,
            epsilon
        )
        

        record_path = save_path + "run_result/"
        mksure_path(record_path)
        

        xline=list(np.arange(len(rewards_list)))
        plt.plot(xline,rewards_list)
        plt.savefig(record_path + 'runresult.svg')

        with open(record_path + "rewards_list.txt", "wb") as fp:  
            pickle.dump(rewards_list, fp)

        time_end = time.time()
        print('time cost', time_end - time_start, 's')


    elif mode == "test":
        datasets_list = load_test_datasets()
        print(len(datasets_list))

        for dataset_num in range(begin, end):
            al_find = 0
            al_re = 0
            al_avgsota = 0
            al_dis = 0

            print(str(dataset_num) + ": ", datasets_list[dataset_num].get_task_names()[0])

            for i in range(test_times):  
                print("*"*8 + "Case " + str(i) + "*"*8)
                alre, alsota, aldis = test(
                    datasets_list,
                    dataset_num,
                    save_path,
                    test_path,
                    mode,
                    lstmdim,
                    search_space,
                    searchtimes, 
                    num_iter,
                    optnum,
                    maxepoch,
                    gamma,
                    pri,
                    active_flag,
                    epsilon
                )
                if aldis < 1e-5:
                    al_find += 1

                al_re += alre
                al_avgsota += alsota
                al_dis += aldis

            print("*"*8 + "Statistic Performance" + "*"*8)
            print("-"*5 + "Active Learning" + "-"*5)
            print("re: ", al_re / test_times)
            print("avgsota: ", al_avgsota / test_times)
            print("distance: ", al_dis / test_times)
            print("find sota cnt: %d in %d searches, ratio: %f" % (al_find, test_times, al_find/test_times))


        time_end = time.time()
        print('time cost', time_end - time_start, 's')



    elif mode == "custom":
        al_find = 0
        al_re = 0
        al_avgsota = 0
        al_dis = 0

        dataset = dc.data.DiskDataset(custom)
        print(dataset)

        test_times = len(dataset)


        for i in range(test_times): 
            print("*"*8 + "Custom " + str(i) + "*"*8)
            alre, alsota, aldis = case(
                dataset,
                save_path,
                test_path,
                mode,
                i,  
                lstmdim,
                search_space,
                searchtimes, 
                num_iter,
                optnum,
                maxepoch,
                gamma,
                pri,
                active_flag,
                epsilon
            )
            if aldis < 1e-5:
                al_find += 1

            al_re += alre
            al_avgsota += alsota
            al_dis += aldis

        print("*"*8 + "Statistic Performance" + "*"*8)
        print("-"*5 + "Active Learning" + "-"*5)
        print("re: ", al_re / test_times)
        print("avgsota: ", al_avgsota / test_times)
        print("distance: ", al_dis / test_times)
        print("find sota cnt: %d in %d searches, ratio: %f" % (al_find, test_times, al_find/test_times))


        time_end = time.time()
        print('time cost', time_end - time_start, 's') 

    else:
        raise Exception("mode error!!!") 

