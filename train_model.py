import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


import argparse

import time


import boto3
import sagemaker
import pandas as pd 
import torch 
import re
import glob
import os
import logging 

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


#objects for dataset download if we need it
boto3_session = boto3.session.Session(region_name='us-east-1')
sagemaker_session = sagemaker.Session(boto3_session)
bucket = 'sagemaker-us-east-1-532709353901'
dataset_prefix = 'dog-breed-data/'
    

from smdebug import modes
from smdebug.pytorch import get_hook

def test(model, test_loader, loss_function, device, hook):
    '''
    This functions conducts the test routine. 
    '''
    
    model.eval()
    
    if hook:
        hook.set_mode(modes.EVAL)
    
    test_loss = 0 
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            outputs = model(data)
            loss = loss_function(outputs, target)
            test_loss += loss.item()
            
    
    average_test_loss = test_loss/len(test_loader)
    
    print(f'Average test loss: {average_test_loss}')
            

def train(model, train_loader, valid_loader, loss_function, optimizer, epochs, device, hook):
    '''
    This function conducts training on the fine tuning network.
    
    It also uses the validation set to report validation statistics as it progresss through each training epoch.

    '''
    epoch_times = []
    for epoch in range(epochs):
        
        model.train()
    
        if hook:
            hook.set_mode(modes.TRAIN)
    
        start = time.time()
    
        training_loss=0
        correct=0
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            pred = model(data)
            loss = loss_function(pred, target)
            training_loss+=loss
            loss.backward()
            optimizer.step()
            pred=pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        
        average_training_loss = training_loss/len(train_loader)
        
        if hook:
            hook.set_mode(modes.EVAL)
        
        model.eval()
        validation_loss = 0
        with torch.no_grad():
            for data, target in valid_loader:
                data = data.to(device)
                target = target.to(device)
                pred = model(data)
                
                loss = loss_function(pred, target)
                validation_loss += loss

        epoch_time = time.time() - start
        epoch_times.append(epoch_time)
        
        average_validation_loss = validation_loss/len(valid_loader)
        
        print(f"Epoch {epoch}: Average training loss {average_training_loss:.4f}, Average validation loss {average_validation_loss:.4f}, in {epoch_time:.2f} sec")

    median_epoch_time =  np.percentile(epoch_times, 50)
    
    print(median_epoch_time)
    
    return model

    
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
    This function creates dataloaders based upon the ImageFolder class in pytorch.
    
    This assumes that the model is being trained on images which are "labelled" by being loaded into the relevant directory.
    
    Transforms applied are applied after consulting pytorch documentation here - https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
    
    '''
    
    
    transform_means = [0.485, 0.456, 0.406]
    transform_stds = [0.229, 0.224, 0.225]
    
    #https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html#torchvision.models.resnet
    transform_train = transforms.Compose(
                                        [
                                            transforms.RandomCrop(256, pad_if_needed = True),
                                            transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=transform_means, std=transform_stds)
                                        ])
    transform_test_valid = transforms.Compose( 
                                             [
                                                 transforms.CenterCrop(256),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=transform_means, std=transform_stds)
                                             ])
    
    
    train_dataset = ImageFolder(root = f'{data_path}/train', transform = transform_train)
    valid_dataset = ImageFolder(root = f'{data_path}/valid', transform = transform_test_valid)
    test_dataset = ImageFolder(root = f'{data_path}/test', transform = transform_test_valid)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size = batch_size, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

    return train_loader, valid_loader, test_loader



def main(args):
    
    #get smdebug logging hook
    hook = get_hook(create_if_not_exists = True)
  
    #initialise model
    model=net()
    
    #loss and optimiser
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters(), args.lr)
    
    #register loss for debug to track.
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
    
    #train
    model = train(model, train_loader, valid_loader, loss_criterion, optimizer, args.epochs, device, hook)
    
    #test
    test(model, test_loader, loss_criterion, device, hook)
    
    #save the model - using state dict 
    model_save_path = 'model.pth'
    torch.save(model.to(torch.device('cpu')).state_dict(), model_save_path)

if __name__=='__main__':
    parser=argparse.ArgumentParser()                 
    
    #for the moment, only intersted in batch_size, epoch and learning rate. 
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type = float, default =1e-3)
    
    args=parser.parse_args()
    

    main(args)
