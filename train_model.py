#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


import argparse



import boto3
import sagemaker
import pandas as pd 
import torch 
import re
import glob
import os
import logging 

#objects for datasset download if we need it
sagemaker_session = sagemaker.Session()
bucket = 'sagemaker-us-east-1-532709353901'
dataset_prefix = 'dog-breed-data/'
    

#TODO: Import dependencies for Debugging andd Profiling

from smdebug import modes
from smdebug.pytorch import get_hook

def test(model, test_loader, loss_function, device):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    
    model.eval()
    
    if hook:
        hook.set_mode(modes.EVAL)
    
    validation_loss = 0 
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = data.to(device)
            outputs = model(data)
            loss = loss_function(outputs, targets)
            val_loss += loss.item()
            

def train(model, train_loader, criterion, optimizer, epochs, device):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.train()
    
    if hook:
        hook.set_mode(modes.TRAIN)
        
    for e in range(epoch):
        running_loss=0
        correct=0
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            pred = model(data)
            loss = criterion(pred, target)
            running_loss+=loss
            criterion.backward()
            optimizer.step()
            pred=pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
def net():
    '''
    Using resnet18 because anything else was frankly painfully large.
    '''
    
    model = models.resnet18(pretrained=True)
    
    #set the gradients to false - do not train the gradients of resnet
    for parameter in model.parameters():
        parameter.requires_grad = False
        
    #fc means something like final fully connected layer - we replace this with the characteristics we want.
    pretrained_output_parameters = model.fc.in_features
    model.fc = nn.Sequential(
                             nn.Linear(pretrained_output_parameters,500),
                             nn.ReLU(),
                             nn.Linear(500, 133)
    )

    return model
    
    
    
def create_data_loaders(data_path, batch_size):
    '''
    
    '''
    
    
    #https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html#torchvision.models.resnet
    transform_train = transforms.Compose(
                                        [
                                            transforms.RandomCrop(256, pad_if_needed = True),
                                            transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.2023, 0.1994, 0.2010])
                                        ])
    transform_test_valid = transforms.Compose( 
                                             [
                                                 transforms.CenterCrop(256),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.2023, 0.1994, 0.2010])
                                             ])
    
    
    train_dataset = ImageFolder(root = './dogImages/train', transform = transform_train)
    valid_dataset = ImageFolder(root = './dogImages/valid', transform = transform_test_valid)
    test_dataset = ImageFolder(root = './dogImages/test', transform = transform_test_valid)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size = batch_size, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

    return train_loader, valid_loader, test_loader

  
    


def main(args):
    
    hook = get_hook(create_if_not_exists = True)
  
    model=net()
    
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters(), args.lr)
    
    if hook:
        hook.register_loss(loss_criterion)
    
    #check if you have the dog images - if not download them
    if not os.path.exists('./dogImages'):
        sagemaker_session.download_data(path = './dogImages', bucket = bucket, key_prefix = dataset_prefix)
    
    #we in gpu land?
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    #push model to device
    model = model.to(device)
    
    #dataset loaders
    train_loader, valid_loader, test_loader = create_data_loaders('./dogImages', args.batch_size)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    train(model, train_loader, loss_criterion, optimizer, args.epoch, device)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_loader, loss_criterion, device)
    
    '''
    TODO: Save the trained model
    '''
    torch.save(model, path)

if __name__=='__main__':
    parser=argparse.ArgumentParser()                 
    
    
    parser.add_argument("--batch_size", type=int, default=)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--lr", type = float, default =1e-3)
    
    args=parser.parse_args()
    

    main(args)
