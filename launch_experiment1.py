#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 18:01:35 2020

@author: michal1
"""
import os
import torch

from experimenter import build_fold, train_test, prepare_db, get_empty_results_table, results_to_tables


class Params():
    pass
params = Params()

# these are fixed params for all experiments
params.num_epoch = 1
params.batch_size = 64
params.num_classes = 2
params.lr = 0.001
params.lr_f = 0.0001
params.db_name = 'NTU-PI-v1-gender'
params.male = 'male'
params.female = 'female'
params.b = 0
params.train_size = 0.7
params.val_fraction = 0.5 # (1-params.train_size)*params.val_fraction = val_size. thus 70%, 15% and 15% for train val test, respectively.
results_folder = 'results_out'

# set up database and result folder
if not os.path.exists(params.db_name):
    print('preparing '+params.db_name +' for experiments')
    prepare_db()

if not os.path.exists(results_folder):
    os.mkdir(results_folder)

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

networks = ['resnet50','densenet121','vgg16','alexnet','squeezenet10']
#networks = ['resnet50','densenet121']
repetitions = ['1', '2', '3', '4', '5']

df_acc, df_tfr, df_tmr = get_empty_results_table(networks,repetitions)
 
# these params are not fixed
imgType = 'imgOrg'
fine_tune = 'all' # 'fc' for finetune only last fc layer, and 'all' for finetune all layers
for model in networks:
    for i in repetitions:
       
        params.seedR = int(i) # repetition random seed
        model_name = model # model name
        
        # build fold for the experiment
        build_fold(params, imgType)
        
        # train and val and then test
        acc_test, true_female_rate, true_male_rate = train_test(params, device, imgType, model_name, fine_tune)
        
        # write experiment results
        df_acc.loc[model,i] = acc_test
        df_tfr.loc[model,i] = true_female_rate
        df_tmr.loc[model,i] = true_male_rate

results_to_tables(df_acc,df_tfr,df_tmr,imgType,fine_tune,results_folder,networks)        


        
