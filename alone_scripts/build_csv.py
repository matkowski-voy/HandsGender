import time
from helpers import buildTainValTestSets, getPathToCSV
import sys
import os


class Params():
    pass
params = Params()

imgType = 'imgOrg'
params.root_dir = os.path.join('NTU-PI-v1-gender',imgType)
params.male = 'male'
params.female = 'female'
params.b = 0
params.train_size = 0.7
params.val_fraction = 0.5 # (1-params.train_size)*params.val_fraction = val_size
params.num_seed = 1



for i in range(params.num_seed):
    params.seedR = i+1
    timestr = time.strftime("%Y%m%d-%H%M%S")
    old_stdout = sys.stdout
    log_file = open(os.path.join(getPathToCSV(params,None),timestr+".log"),"w")
    sys.stdout = log_file
    start = time.time()
    
    print('----------------------------------------------------------------------')
    print('repetition random seed->'+str(params.seedR))
    print('-building train, val, test csv files')
    buildTainValTestSets(params)
    
    end = time.time()
    print(end-start)
    print('----------------------------------------------------------------------')
    sys.stdout = old_stdout
    log_file.close()
   
