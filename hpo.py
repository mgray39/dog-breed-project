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



def test(model, test_loader):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    pass

def train(model, train_loader, criterion, optimizer):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    pass
    
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

def create_data_loaders(data_dir, batch_size):

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
    
    
    train_dataset = ImageFolder(root = f'{data_dir}/train', transform = transform_train)
    valid_dataset = ImageFolder(root = f'{data_dir}/valid', transform = transform_test_valid)
    test_dataset = ImageFolder(root = f'{data_dir}/test', transform = transform_test_valid)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size = batch_size, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

    return train_loader, valid_loader, test_loader


def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = None
    optimizer = None
    
    
    #dataset loaders - according to this - https://docs.aws.amazon.com/sagemaker/latest/dg/model-access-training-data.html
    
    #data is copied to
    #/opt/ml/input/data/training-channel
    train_loader, valid_loader, test_loader = create_data_loaders('/opt/ml/input/data/training-channel', args.batch_size)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    
    so according to this - https://docs.aws.amazon.com/sagemaker/latest/dg/model-access-training-data.html
    
    data is copied to
    /opt/ml/input/data/training-channel
    '''
    
    
    
    model=train(model, train_loader, loss_criterion, optimizer)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_loader, criterion)
    
    '''
    TODO: Save the trained model
    '''
    torch.save(model, path)

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )
    
    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    main(parser.parse_args())