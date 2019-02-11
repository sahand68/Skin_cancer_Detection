
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

# In[2]:



def load_model( num_labels=3):


    model =models.resnet152(pretrained=True)

    n_class = 3


    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, n_class)

    #chpt = torch.load(checkpoint_path)
    #model.class_to_idx = chpt['class_to_idx']
    #model.load_state_dict(chpt['state_dict'])
    ct = 0
    for name, child in model.named_children():
        ct += 1
        if ct < 7:
            for name2, params in child.named_parameters():
                params.requires_grad = False


    return model

def train_model(model, hidden_units , epochs, learning_rate, device , image_path):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    since =time.time()
    criterion = nn.CrossEntropyLoss()
    optimizer= optim.Adam(model.parameters(), lr=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1, last_epoch=-1) ## dynamic learning rate!

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0



    print('Number of hidden units:', hidden_units)
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


def check_accuracy_on_test(testloader, checkpoint_path, loaded_model):
    correct = 0
    total = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model =loaded_model
    model.cuda()


    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images= images.to(device)
            labels = labels.to(device)
            chpt = torch.load(checkpoint_path)
            model.class_to_idx = chpt['class_to_idx']
            model.load_state_dict(chpt['state_dict'])
            model.eval()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))


# In[5]:


def process_image(image_test):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array'''
    img_loader = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation((-180, 180)),
        transforms.ToTensor()])

    pil_image = Image.open(image_test)
    pil_image = img_loader(pil_image).float()

    tensor_image = np.array(pil_image)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    tensor_image = (np.transpose(tensor_image, (1, 2, 0)) - mean)/std
    tensor_image = np.transpose(tensor_image, (2, 0, 1))


    return tensor_image


# In[6]:



def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    if title:
        plt.title(title)
    # PyTorch tensors assume the color channel is first
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax

# In[7]:


def predict(image_test,model,checkpoint_path, topk = 5):



    image = process_image(image_test)
    image_tensor = torch.from_numpy(image).type(torch.FloatTensor)
    input= image_tensor.unsqueeze(0)



    chpt = torch.load(checkpoint_path)

    model.class_to_idx = chpt['class_to_idx']
    model.load_state_dict(chpt['state_dict'])

    model.eval()
    probs = torch.exp(model.forward(input))
    top_probs, top_labs = probs.topk(topk)
    top_probs = top_probs.detach().numpy().tolist()[0]
    top_labs = top_labs.detach().numpy().tolist()[0]

    idx_to_class = {val: key for key, val in model.class_to_idx.items()}

    top_labels = [idx_to_class[lab] for lab in top_labs]
    top_flowers = [Dermo_cat[idx_to_class[lab]] for lab in top_labs]
    return  top_probs, top_labels, top_flowers



# In[8]:






# In[9]:



def generate_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, action="store", default="flowers")
    parser.add_argument('--model_path', dest="model_path", action="store", default='vgg16.pth.tar')
    parser.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
    parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=4096)
    parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
    parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=25)
    parser.add_argument('--device', dest='device', action="store", default="gpu")

    return parser.parse_args()


# In[13]:






image_path = "C://Users//Sahan//ipthw//dermatologist-ai//data"
arch = 'ResNet-152'
hidden_units = 4096
epochs=25
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
learning_rate =0.001



print(device)


loaded_model =load_model(num_labels=3)




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
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,shuffle=True) for x in ['train', 'valid','test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid','test']}
class_names = image_datasets['train'].classes
trained_model =train_model(loaded_model, hidden_units , epochs, learning_rate, device, image_path)

trained_model.class_to_idx = image_datasets['train'].class_to_idx
checkpoint = {
        'arch': arch,
        'class_to_idx': trained_model.class_to_idx,
        'state_dict': trained_model.state_dict(),
        'hidden_units': 4096}

torch.save(checkpoint, 'ResNet-152.pth.tar')
checkpoint_path = 'ResNet-152.pth.tar'
image_test = 'C:\\Users\\sahan\\ipthw\\dermatologist-ai\\train\\melanoma\\ISIC_0000002.jpg'

check_accuracy_on_test(dataloaders['test'], checkpoint_path,loaded_model)

top_probs, top_labels, top_flowers =predict(image_test,loaded_model, checkpoint_path, topk=5)
