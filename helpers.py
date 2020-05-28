# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 12:41:28 2020

@author: MATK0001
"""

import os
import numpy as np
import random
import torch
import shutil
import torch.nn as nn
from torchvision import models



def listFilesfromDB(root_dir,dbName,listSet):
    setFiles = [os.path.join(dbName,i,j) for i in listSet for j in os.listdir(os.path.join(root_dir,dbName, i))]
    return setFiles

def listFilesfromDB2(root_dir,dbName,listSet):
    setFiles = [os.path.join(dbName,i) for i in listSet]
    return setFiles

def getListOfDisjointIDs(root_dir,dbName,train_size,val_fraction,seedR):
    
    random.seed(seedR)
    
    listOfFile = os.listdir(os.path.join(root_dir,dbName))
    
           
    # shuffle IDs 
    print('---shuffling list of IDs in->'+dbName)
    random.shuffle(listOfFile)
    train_size = int(train_size * len(listOfFile))
        
    listTrain =  listOfFile[0:train_size]
    listTestTmp = listOfFile[train_size:]
    test_size = int(val_fraction * len(listTestTmp))
    listVal = listTestTmp[0:test_size]
    listTest = listTestTmp[test_size:]
    
    trainFiles = list()
    valFiles = list()
    testFiles = list()
    
    print('---listing train files')
    trainFiles=listFilesfromDB(root_dir,dbName,listTrain)
    
    print('---listing val files')  
    valFiles=listFilesfromDB(root_dir,dbName,listVal)
    
    print('---listing test files') 
    testFiles=listFilesfromDB(root_dir,dbName,listTest)
                     
    return trainFiles, valFiles, testFiles

def getPathToCSV(params,set_p):
    
    root_dir=params.root_dir
    female=params.female
    male=params.male
    seedR=params.seedR 
    b = params.b
    val_set_size = (1-params.train_size)*params.val_fraction*100
    test_set_size = (1-params.train_size-val_set_size/100)*100
    csv_fold_name = os.path.join('sets'
                                 +str(int(params.train_size*100))+'_'
                                 +str(int(val_set_size))+'_'
                                 +str(int(test_set_size))+'_'
                                 +str(b).replace(".",""),
                                 "Class1-"+female+"_"+"Class0-"+male
                                 ,str(seedR))
    if seedR == 'all':
        pass
    else:
        if not os.path.exists(os.path.join(root_dir,csv_fold_name)):
                os.makedirs(os.path.join(root_dir,csv_fold_name))
    if set_p is not None:
        csv_file_name = set_p+"_set.csv"        
        return os.path.join(root_dir,csv_fold_name,csv_file_name)
    else:
        return os.path.join(root_dir,csv_fold_name)

def checkIfCSVexist(params,train_set,val_set,test_set):
    
    root_dir=params.root_dir
    female=params.female
    male=params.male
    seedR=params.seedR 
    b = params.b
    
    train_set_exist = False
    test_set_exist = False
    val_set_exist = False
    sets_exist = False
    
    val_set_size = (1-params.train_size)*params.val_fraction*100
    test_set_size = (1-params.train_size-val_set_size/100)*100
    csv_fold_name = os.path.join('sets'
                             +str(int(params.train_size*100))+'_'
                             +str(int(val_set_size))+'_'
                             +str(int(test_set_size))+'_'
                             +str(b).replace(".",""),
                             "class1-"+female+"_"+"class2-"+male
                             ,str(seedR))
    csv_file_name = train_set+"_set.csv"
    if os.path.exists(os.path.join(root_dir,csv_fold_name,csv_file_name)):
        print('already exists->'+os.path.join(root_dir,csv_fold_name,csv_file_name))
        train_set_exist = True
    else:
        print('do not exists->'+os.path.join(root_dir,csv_fold_name,csv_file_name))
    
    csv_file_name = val_set+"_set.csv"
    if os.path.exists(os.path.join(root_dir,csv_fold_name,csv_file_name)):
        print('already exists->'+os.path.join(root_dir,csv_fold_name,csv_file_name))
        val_set_exist = True
    else:
        print('do not exists->'+os.path.join(root_dir,csv_fold_name,csv_file_name))
        
    csv_file_name = test_set+"_set.csv"
    if os.path.exists(os.path.join(root_dir,csv_fold_name,csv_file_name)):
        print('already exists->'+os.path.join(root_dir,csv_fold_name,csv_file_name))
        test_set_exist = True
    else:
        print('do not exists->'+os.path.join(root_dir,csv_fold_name,csv_file_name))
    
    if (train_set_exist == True and val_set_exist == True and test_set_exist == True):
        sets_exist = True
        
    return sets_exist
        
    

def buildTainValTestSets(params):
    
    root_dir=params.root_dir
    female=params.female
    male=params.male
    train_size=params.train_size
    val_fraction=params.val_fraction
    seedR=params.seedR 
    b = params.b
    random.seed(seedR)
    
    # set 
    print('--building male classes')
    train0,val0,test0 = getListOfDisjointIDs(root_dir,male,train_size,
                                             val_fraction,seedR)
    
    # set
    print('--building female classes')
    train1,val1,test1 = getListOfDisjointIDs(root_dir,female,train_size,
                                             val_fraction,seedR)
    
    
    
    if(len(train0)*b > len(train1)):
        print('---taking random train subset from male classes')
        train0 = random.sample(train0, round(len(train1)/b))
           
    if(len(train1)*b > len(train0)):
        print('---taking random train subset from female classes')
        train1 = random.sample(train1, round(len(train0)/b))
    
    label0=[0]*len(train0)    
    label1=[1]*len(train1)
    ratioS2U = len(train1)/len(train0)
    print('[female to male] ratio->',ratioS2U)
    np.savetxt(os.path.join(getPathToCSV(params,None),'ratio_train.txt'),[ratioS2U], fmt='%s')
    labelTrain=label0 + label1
    #random.shuffle(labelTrain) #only to test overfit
    
    set_p = 'train'
    path_save = getPathToCSV(params,set_p)
    np.savetxt(path_save, np.column_stack((train0 + train1, labelTrain)),
               delimiter=",", fmt='%s')
   
    print('--train set saved in:' + path_save)
    print('---train set number of samples:',str(len(train0)+len(train1)))
    print('---train set number of male samples:',str(len(train0)))
    print('---train set number of female samples:',str(len(train1)))
      
    
    if(len(val0)*b > len(val1)):
        print('---taking random val subset from male classes')
        val0 = random.sample(val0, len(val1))
        
    if(len(val1)*b > len(val0)):
        print('---taking random val subset from female classes')
        val1 = random.sample(val1, len(val0))
            
    label0=[0]*len(val0)
    label1=[1]*len(val1)
    
    set_p = 'val'
    path_save = getPathToCSV(params,set_p)
    np.savetxt(path_save, np.column_stack((val0 + val1, label0 + label1)),
               delimiter=",", fmt='%s')
    print('--val set saved in:' + path_save)
    print('---val set number of samples:',str(len(val0)+len(val1)))
    print('---val set number of male samples:',str(len(val0)))
    print('---val set number of female samples:',str(len(val1)))
    
    
    if(len(test0)*b > len(test1)):
        print('---taking random test subset from male classes')
        test0 = random.sample(test0, len(test1))    
        
    if(len(test1)*b > len(test0)):
        print('---taking random test subset from female classes')
        test1 = random.sample(test1, len(test0))
    
    label0=[0]*len(test0)
    label1=[1]*len(test1)
    
    set_p = 'test'
    path_save = getPathToCSV(params,set_p)
    np.savetxt(path_save, np.column_stack((test0 + test1, label0 + label1)),
               delimiter=",", fmt='%s')
    print('--test set saved in:' + path_save)
    print('---test set number of samples:',str(len(test0)+len(test1)))
    print('---test set number of male samples:',str(len(test0)))
    print('---test set number of female samples:',str(len(test1)))

    
def path_to_saved_model(params,modelType):
    if not os.path.exists(os.path.join(getPathToCSV(params,None),modelType)):
        os.mkdir(os.path.join(getPathToCSV(params,None),modelType))
    return os.path.join(getPathToCSV(params,None),modelType)

def path_to_saved_model_results(params,modelType):
    if not os.path.exists(os.path.join(getPathToCSV(params,None),modelType,'results')):
        os.mkdir(os.path.join(getPathToCSV(params,None),modelType,'results'))
    return os.path.join(getPathToCSV(params,None),modelType,'results')

def save_checkpoint(state, is_best, params, modelType):
    filename=os.path.join(path_to_saved_model(params,modelType),'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(path_to_saved_model(params,modelType),'model_best.pth.tar'))
        print('saving best model')

 
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):

    model_ft = None

    if model_name == "resnet50":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)

    elif model_name == "vgg16":
        """ VGG11_bn
        """
        model_ft = models.vgg16(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)

    elif model_name == "squeezenet10":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes

    elif model_name == "densenet121":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)

    else:
        print("Invalid model name, exiting...")
#        exit()

    return model_ft