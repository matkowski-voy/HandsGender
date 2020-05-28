import os
import torch
from torchvision import transforms
import random
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.optim as optim
from helpers import getPathToCSV, path_to_saved_model_results, save_checkpoint, path_to_saved_model, initialize_model
from dataloader.dataloader import ImageFromCSVLoader
import time
from helpers import buildTainValTestSets
import sys
import shutil


def train_test(params, device, imgType, model_name, fine_tune):
    
    params.root_dir = os.path.join(params.db_name,imgType)
    print('---------------------------')
    print('image type: '+imgType)
    if fine_tune == 'fc':
        print('finetune only last fc layer')
        feature_extract = True
        
    if fine_tune == 'all':
        print('finetune all layers')
        feature_extract = False    
       
    random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    transform_train=transforms.Compose([transforms.Resize((224,224)),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.RandomAffine((-90,90), translate=(.1,.1), scale=(0.9,1.1)),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]) 
        
    transform=transforms.Compose([transforms.Resize((224,224)),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])        
    
    trainset = ImageFromCSVLoader(root_dir=params.root_dir, csv_file=getPathToCSV(params,'train'), transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=params.batch_size,
                                            shuffle=True, num_workers=0)
    
    valset = ImageFromCSVLoader(root_dir=params.root_dir, csv_file=getPathToCSV(params,'val'), transform=transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=params.batch_size,
                                            shuffle=True, num_workers=0)
    
    testset = ImageFromCSVLoader(root_dir=params.root_dir, csv_file=getPathToCSV(params,'test'), transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=params.batch_size,
                                            shuffle=True, num_workers=0)
    
    # Initialize 
    net = initialize_model(model_name, params.num_classes, feature_extract, use_pretrained=True)
    
    
    criterion = nn.CrossEntropyLoss()
    
    if fine_tune == 'all':
        optimizer = optim.Adam([{'params':list(net.parameters())[:-1], 'lr': params.lr_f},
                                 {'params': list(net.parameters())[-1], 'lr': params.lr}])
    if fine_tune == 'fc':
        optimizer = optim.Adam([{'params': list(net.parameters())[-1], 'lr': params.lr}])
    
    net.to(device)
    
    # train and val
    print('Start Training')
    print('network: '+model_name)
    print('repetition: '+str(params.seedR))
    best_acc_val = 0.0
    for epoch in range(params.num_epoch):  # loop over the dataset multiple times
    
        running_loss = 0.0
        correct = 0
        total = 0
        for i, batch in enumerate(trainloader,0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = batch['image'].to(device), batch['label'].to(device)
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            net.train()
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            # print statistics
            running_loss += loss.item()
            if i % 20 == 19:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2))
                running_loss = 0.0
                
        print('Acc train images: %.2f %%' % (
            100 * correct / total))
        
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for batch in valloader:
                images, labels = batch['image'].to(device), batch['label'].to(device)
                net.eval()
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        acc_val = (100 * correct_val / total_val)
        print('Acc val images: %.2f %%' % acc_val)
        
        is_best = acc_val > best_acc_val
        best_acc_val = max(acc_val, best_acc_val)
    
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'best_acc1': best_acc_val,
            'optimizer' : optimizer.state_dict(),
        }, is_best, params, model_name)
    
    print('Finished Training')
    
    # load and test
    checkpoint = torch.load(os.path.join(path_to_saved_model(params,model_name),'model_best.pth.tar'))
    net.load_state_dict(checkpoint['state_dict'])
    
    correct_test = 0
    total_test = 0
    true_female = 0
    true_male = 0
    false_female = 0
    false_male = 0
    outputsT = []
    labelsT = []
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch['image'].to(device), batch['label'].to(device)
            net.eval()
            outputs = net(images)
            
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
            true_female += (predicted[labels==1] == labels[labels==1]).sum().item()
            true_male += (predicted[labels==0] == labels[labels==0]).sum().item()
            false_female += (predicted[labels==1] != labels[labels==1]).sum().item()
            false_male += (predicted[labels==0] != labels[labels==0]).sum().item()
            
            outputsT.append(outputs.detach().cpu().numpy().squeeze())
            labelsT.append(labels.to(dtype=float).detach().cpu().numpy())
            
    acc_test = (100 * correct_test / total_test)
    true_female_rate = (100 * true_female/ (true_female+false_male))
    true_male_rate = (100 * true_male/ (true_male+false_female))
    print('Acc test images: %.2f %%' % acc_test)
    
    np.save(os.path.join(path_to_saved_model_results(params,model_name),'scores'), outputsT)
    np.save(os.path.join(path_to_saved_model_results(params,model_name),'targets'), labelsT)
    
    return acc_test, true_female_rate, true_male_rate

def get_empty_results_table(networks,repetitions):
    df_acc = pd.DataFrame(data=None, index=networks, columns=repetitions)
    df_tfr = pd.DataFrame(data=None, index=networks, columns=repetitions)
    df_tmr = pd.DataFrame(data=None, index=networks, columns=repetitions)
    
    return df_acc, df_tfr, df_tmr

def results_to_tables(df_acc,df_tfr,df_tmr,imgType,fine_tune,results_folder,networks):
    name = '-imgTpye-'+imgType+'-finetune-'+fine_tune+'.csv'
    df_acc.to_csv(os.path.join(results_folder,'acc'+name))
    df_tfr.to_csv(os.path.join(results_folder,'tfr'+name))
    df_tmr.to_csv(os.path.join(results_folder,'tmr'+name))
    print('accuracy\n',df_acc)
    print('------------------------------------------------------------------')
    print('true female rate\n',df_tfr)
    print('------------------------------------------------------------------')
    print('true male rate\n',df_tmr)
    print('------------------------------------------------------------------')
    df_tab = pd.DataFrame(data=None, index=networks, columns=['mean acc','std acc','mean tfr','std tfr','mean tmr','std tmr'])
    df_tab.loc[:,'mean acc'] = df_acc.mean(axis=1)
    df_tab.loc[:,'std acc'] = df_acc.std(axis=1)
    df_tab.loc[:,'mean tfr'] = df_tfr.mean(axis=1)
    df_tab.loc[:,'std tfr'] = df_tfr.std(axis=1)
    df_tab.loc[:,'mean tmr'] = df_tmr.mean(axis=1)
    df_tab.loc[:,'std tmr'] = df_tmr.std(axis=1)
    print('table mean and std\n',df_tab)
    print('------------------------------------------------------------------')
    df_tab.to_csv(os.path.join(results_folder,'mean-std'+name))    

def build_fold(params,imgType):
    params.root_dir = os.path.join(params.db_name,imgType)
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

# this function rebuilds NTU-PI-v1 databse to male/female structure
# NTU-PI-v1-gender will contain the resultant database and other intermediate results
# such as images, csv train val test files, repetitions foe experiments, saved best newtworks, etc.
# Thus NTU-PI-v1-gender is not a database. it is a processed and saved NTU-PI-v1 in a different form
# based on provided by me gender labeles. It is the folder which containes the results of a processed
# NTU-PI-v1
def prepare_db():

    gender_csv_file_name = 'NTU-PI-v1-gender.csv'
    dst_db_name = 'NTU-PI-v1-gender'
    src_db_name = 'NTU-PI-v1'
    
    if not os.path.exists('NTU-PI-v1'):
        print('download NTU-PI-v1 database and extract here. See how to obtain database: https://github.com/matkowski-voy/Palmprint-Recognition-in-the-Wild')
        
    
    data = pd.read_csv(gender_csv_file_name,header=None)
    
    if not os.path.exists(dst_db_name):
        os.mkdir(dst_db_name)
        os.mkdir(os.path.join(dst_db_name,'imgOrg'))
        os.mkdir(os.path.join(dst_db_name,'imgOrg','male'))
        os.mkdir(os.path.join(dst_db_name,'imgOrg','female'))
        os.mkdir(os.path.join(dst_db_name,'imgMask'))
        os.mkdir(os.path.join(dst_db_name,'imgMask','male'))
        os.mkdir(os.path.join(dst_db_name,'imgMask','female'))
        
        # only using landmarks because NTU-PI-v1 does not provide ROIs
        if os.path.exists(os.path.join(src_db_name,'flip','ROImarked')):
            os.mkdir(os.path.join(dst_db_name,'ROImarked'))
            os.mkdir(os.path.join(dst_db_name,'ROImarked','male'))
            os.mkdir(os.path.join(dst_db_name,'ROImarked','female'))
    
    for i in range(len(data)):
        
        subject_id = data.iloc[i,0][0:data.iloc[i,0].find('-')]
        subject_gender = data.iloc[i,1]
        
        # imgOrg
        img_type = 'imgOrg'
        if not os.path.exists(os.path.join(dst_db_name,img_type,subject_gender,subject_id)):
            os.mkdir(os.path.join(dst_db_name,img_type,subject_gender,subject_id))
            
        if os.path.exists(os.path.join(src_db_name,'flip',img_type,'train',data.iloc[i,0])):
            shutil.copyfile(os.path.join(src_db_name,'flip',img_type,'train',data.iloc[i,0]), \
                            os.path.join(dst_db_name,img_type,subject_gender,subject_id,data.iloc[i,0]))
        
        if os.path.exists(os.path.join(src_db_name,'flip',img_type,'test',data.iloc[i,0])):
            shutil.copyfile(os.path.join(src_db_name,'flip',img_type,'test',data.iloc[i,0]), \
                            os.path.join(dst_db_name,img_type,subject_gender,subject_id,data.iloc[i,0]))
                
        # imgMask
        if not os.path.exists(os.path.join(dst_db_name,'imgMask',subject_gender,subject_id)):
            os.mkdir(os.path.join(dst_db_name,'imgMask',subject_gender,subject_id))
        
        if os.path.exists(os.path.join(src_db_name,'imgMask',data.iloc[i,0][0:data.iloc[i,0].find('C')-1]+'.jpg')):
            
            shutil.copyfile(os.path.join(src_db_name,'imgMask',data.iloc[i,0][0:data.iloc[i,0].find('C')-1]+'.jpg'), \
                            os.path.join(dst_db_name,'imgMask',subject_gender,subject_id,data.iloc[i,0][0:data.iloc[i,0].find('C')-1]+'.jpg'))
        
        if os.path.exists(os.path.join(src_db_name,'imgMask',data.iloc[i,0][0:data.iloc[i,0].find('C')-1]+'.jpeg')):
            
            shutil.copyfile(os.path.join(src_db_name,'imgMask',data.iloc[i,0][0:data.iloc[i,0].find('C')-1]+'.jpeg'), \
                            os.path.join(dst_db_name,'imgMask',subject_gender,subject_id,data.iloc[i,0][0:data.iloc[i,0].find('C')-1]+'.jpeg'))
    
        # ROImarked
        img_type = 'ROImarked'
        if os.path.exists(os.path.join(src_db_name,'flip','ROImarked')):
            if not os.path.exists(os.path.join(dst_db_name,img_type,subject_gender,subject_id)):
                os.mkdir(os.path.join(dst_db_name,img_type,subject_gender,subject_id))
                
            if os.path.exists(os.path.join(src_db_name,'flip',img_type,'train',data.iloc[i,0])):
                shutil.copyfile(os.path.join(src_db_name,'flip',img_type,'train',data.iloc[i,0]), \
                                os.path.join(dst_db_name,img_type,subject_gender,subject_id,data.iloc[i,0]))
            
            if os.path.exists(os.path.join(src_db_name,'flip',img_type,'test',data.iloc[i,0])):
                shutil.copyfile(os.path.join(src_db_name,'flip',img_type,'test',data.iloc[i,0]), \
                                os.path.join(dst_db_name,img_type,subject_gender,subject_id,data.iloc[i,0]))