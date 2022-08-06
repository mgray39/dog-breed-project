

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import os
from PIL import Image
import io
import sys
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))



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
    
    
JPEG_CONTENT_TYPE = 'image/jpeg'


def model_fn(model_dir):
    
    #we in gpu land?
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    model = net() 
    
    
    with open(os.path.join(model_dir, "model.pth"), "rb") as f: 
        model.load_state_dict(torch.load(f))
    
    #push to whatever the hell device you want 
    model.to(device)
    
    #set to evaluate mode
    model.eval()
    return model

def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
    
    if content_type == JPEG_CONTENT_TYPE:
        image = Image.open(io.BytesIO(request_body))
        
        return image
        
    else:
        print('pong....')
        logger.critical(f'unsupported content type: {content_type}')
    

# inference
def predict_fn(input_object, model):
    
    #we in gpu land?
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    transform_means = [0.485, 0.456, 0.406]
    transform_stds = [0.229, 0.224, 0.225]
    
    transform_test_valid = transforms.Compose( 
                                             [
                                                 transforms.CenterCrop(256),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=transform_means, std=transform_stds)
                                             ])

    
    input_object=transform_test_valid(input_object)
    
    input_object = input_object.to(device)
    
    with torch.no_grad():
        prediction = model(input_object.unsqueeze(0))
    return prediction
