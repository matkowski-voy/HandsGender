import os
import torch
from torchvision import transforms
import random
import torch.nn as nn
import numpy as np
import torch.optim as optim
from helpers import getPathToCSV, save_checkpoint, path_to_saved_model, initialize_model
from dataloader.dataloader import ImageFromCSVLoader

class Params():
    pass
params = Params()



imgType = 'imgOrg'
params.root_dir = os.path.join('NTU-PI-v1-gender',imgType)
params.male = 'male'
params.seen_exclude_id = []
params.female = 'female'
params.unseen_exclude_id = []
params.b = 0
params.train_size = 0.7
params.val_fraction = 0.5 # (1-params.train_size)*params.val_fraction = val_size
params.num_seed = 1
params.seedR = 1

batch_size = 64
feature_extract = False
num_classes = 2
model_name = 'densenet121'

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                        shuffle=True, num_workers=0)

valset = ImageFromCSVLoader(root_dir=params.root_dir, csv_file=getPathToCSV(params,'val'), transform=transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                        shuffle=True, num_workers=0)

testset = ImageFromCSVLoader(root_dir=params.root_dir, csv_file=getPathToCSV(params,'test'), transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                        shuffle=True, num_workers=0)

# Initialize 
net = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)


criterion = nn.CrossEntropyLoss()
  
optimizer = optim.Adam([{'params':list(net.parameters())[:-1], 'lr': 0.0001},
                         {'params': list(net.parameters())[-1], 'lr': 0.001}])

net.to(device)

# train and val
print('Start Training')
best_acc_val = 0.0
for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    correct = 0
    correct2 = 0
    correct3 = 0
    correct22 = 0
    correct33 = 0
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
        correct2 += (predicted[labels==1] == labels[labels==1]).sum().item()
        correct3 += (predicted[labels==0] == labels[labels==0]).sum().item()
        correct22 += (predicted[labels==1] != labels[labels==1]).sum().item()
        correct33 += (predicted[labels==0] != labels[labels==0]).sum().item()
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
with torch.no_grad():
    for batch in testloader:
        images, labels = batch['image'].to(device), batch['label'].to(device)
        net.eval()
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()
acc_test = (100 * correct_test / total_test)
print('Acc on the test images: %.2f %%' % acc_test)


