
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
import copy
import argparse
import os
from PIL import Image
import numpy as np
import json
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

def load_model(arch , checkpoint_path, num_labels=3):


    model =models.resnet152(pretrained=True)

    n_class = 3


    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, n_class)
    chpt = torch.load(checkpoint_path)
    model.class_to_idx = chpt['class_to_idx']
    model.load_state_dict(chpt['state_dict'])
    ct = 0
    for name, child in model.named_children():
        ct += 1
        if ct < 7:
            for name2, params in child.named_parameters():
                params.requires_grad = False


    return model


# In[3]:


def mixup_batch(inp_batch, alpha):
    """
    Applies mixup augementation to a batch
    :param input_batch: tensor with batchsize as first dim
    :param alpha: lamda drawn from beta(alpha+1, alpha)
    """
    inp_clone = inp_batch.clone()
    #getting batch size
    batchsize = inp_batch.size()[0]
     #permute a clone
    perm = np.random.permutation(batchsize)
    for i in range(batchsize):
        inp_clone[i] = inp_batch[perm[i]]
    #generating different lambda for each sample
    #Refernced from http://www.inference.vc/mixup-data-dependent-data-augmentation/
    lam = torch.Tensor(np.random.beta(alpha+1, alpha, batchsize))
    lam = lam.view(-1,1,1,1)
    inp_mixup = lam * inp_batch + (1- lam) * inp_clone
    return inp_mixup


def train_model(model, epochs, learning_rate, device , image_path):



    since =time.time()
    criterion = nn.CrossEntropyLoss()
    optimizer= optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.05, amsgrad=True)

    scheduler =  lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    best_acc = 0.0




    print('Number of epochs:', epochs)
    print('Learning rate:', learning_rate)




    for epoch in range(epochs):

        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)


        for phase in ['train','valid']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            corrects = 0



            for inputs, labels in dataloaders[phase]:

                model.to(device)
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)


                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)


            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())


        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)


    return model

# In[4]:


input_shape = 299
batch_size = 32
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
scale = 360
input_shape = 299
use_parallel = True
use_gpu = True
epochs = 20


def check_accuracy_on_test(testloader, checkpoint_path, loaded_model):
    correct = 0
    total = 0
    model =loaded_model
    chpt = torch.load(checkpoint_path)
    model.class_to_idx = chpt['class_to_idx']
    model.load_state_dict(chpt['state_dict'])

    model.cuda()


    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images= images.to(device)
            labels = labels.to(device)

            model.eval()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))




if __name__ == '__main__':
    image_path = "C://Users//Sahan//ipthw//dermatologist-ai//data"


    data_transforms = {
                'train': transforms.Compose([
                    transforms.RandomRotation(45),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                 ]),
                'valid': transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                 ]),
                'test': transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                 ])}



    image_datasets = {x: datasets.ImageFolder(os.path.join(image_path, x),data_transforms[x]) for x in ['train', 'valid','test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size= 64,shuffle=True,  num_workers=16) for x in ['train', 'valid','test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid','test']}
    class_names = image_datasets['train'].classes


    arch = 'resnet152'


    learning_rate =0.0001

    epochs = 30

    checkpoint_path = 'resnet152-derm.pth.tar'
    model =load_model(arch ,checkpoint_path, num_labels=3)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trained_model =train_model(model, epochs, learning_rate, device, image_path)
    trained_model.class_to_idx = image_datasets['train'].class_to_idx
    checkpoint = {
        'arch': arch,
        'class_to_idx': trained_model.class_to_idx,
        'state_dict': trained_model.state_dict(),
        'hidden_units':1000}

    torch.save(checkpoint, 'resnet152-derm.pth.tar')


    check_accuracy_on_test(dataloaders['test'], checkpoint_path, model)
