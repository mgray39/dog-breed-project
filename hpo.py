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
import torch 
import re
import glob
import os
import logging 
import sys
import json
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, test_loader, loss_function, device):
    '''
    This functions conducts the test routine. 
    '''
    
    model.eval()
    
    test_loss = 0 
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = data.to(device)
            outputs = model(data)
            loss = loss_function(outputs, targets)
            test_loss += loss.item()
            
    
    average_test_loss = test_loss/len(test_loader)
    
    logger.info(f'Average test loss: {average_test_loss}')


def train(model, train_loader, valid_loader, loss_function, optimizer, epochs, device):
    '''
    This function conducts training on the fine tuning network.
    
    It also uses the validation set to report validation statistics as it progresss through each training epoch.

    '''
    epoch_times = []
    for epoch in range(epochs):
        
        model.train()
    
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
        
        logger.info(f"Epoch {epoch}: Average train loss {average_training_loss:.4f}, Average validation loss {average_validation_loss:.4f}, in {epoch_time:.2f} sec")

    median_epoch_time =  np.percentile(epoch_times, 50)
    
    print(median_epoch_time)
    
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
    logger.info('Getting Data Loaders')
        
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
    
    logger.info('Reading data...')
    
    train_dataset = ImageFolder(root = f'{data_dir}/train', transform = transform_train)
    valid_dataset = ImageFolder(root = f'{data_dir}/valid', transform = transform_test_valid)
    test_dataset = ImageFolder(root = f'{data_dir}/test', transform = transform_test_valid)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size = batch_size, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

    return train_loader, valid_loader, test_loader


def main(args):
    logger.info('Initialising network...')
    model=net()
        
    #loss and optimiser
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters(), args.lr)
    
    #we in gpu land?
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    #push model to device
    model = model.to(device)
    
    #dataset loaders - according to this - https://docs.aws.amazon.com/sagemaker/latest/dg/model-access-training-data.html
    #data is copied to
    #/opt/ml/input/data/training-channel
    train_loader, valid_loader, test_loader = create_data_loaders(args.data_dir, args.batch_size)
    
    model=train(model, train_loader, valid_loader, loss_criterion, optimizer, args.epochs, device)
    
    test(model, test_loader, loss_criterion, device)
    
    #save the model - using state dict 
    model_save_path = os.path.join(args.model_dir, 'model.pth')
    
    torch.save(model.to(torch.device('cpu')).state_dict(), model_save_path)
    

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
    
    args = parser.parse_args()
    
    #for key, value in args.items():
    #    logger.info(f'{key}:{value}')

    main(args)