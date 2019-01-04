import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

BATCH_SIZE = 64

def get_image_data(data_dir):
    '''
    Helper method to return datasets and dataloaders for train, test and validation
    from the data directory using ImageFolder
    '''
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'    
    data_transforms = {}
    data_transforms['train'] = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(), 
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    data_transforms['eval'] = transforms.Compose([ transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(), 
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    image_datasets = {}
    image_datasets['train'] = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
    image_datasets['valid'] = datasets.ImageFolder(valid_dir, transform=data_transforms['eval'])
    image_datasets['test'] = datasets.ImageFolder(test_dir, transform=data_transforms['eval'])

    dataloaders = {}
    dataloaders['train'] = DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True)
    dataloaders['valid'] = DataLoader(image_datasets['valid'], batch_size=BATCH_SIZE)
    dataloaders['test'] = DataLoader(image_datasets['test'], batch_size=BATCH_SIZE)
    return image_datasets, dataloaders

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    # Ref - https://stackoverflow.com/questions/273946/how-do-i-resize-an-image-using-pil-and-maintain-its-aspect-ratio
    min_dim = min(image.size)
    thumbnail_size = [x * 256 / min_dim for x in image.size]
    image.thumbnail(thumbnail_size, Image.ANTIALIAS)
    left_x, right_x = int((image.size[0] - 224)/2), int((image.size[0] + 224) / 2)
    top_y, bottom_y = int((image.size[1] - 224)/2), int((image.size[1] + 224) / 2)
    image = image.crop((left_x, top_y, right_x, bottom_y))
    np_image = np.array(image)  
    np_image = np_image / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean ) / std
    return np_image.transpose((2, 0, 1)) 

def save_checkpoint(model, optimizer, 
                    model_name, optimizer_name, criterion_name, 
                    checkpoint_name='trained_model.pth'):
    '''
    Save model and optimizer state and attributes into a checkpoint file
    '''
    checkpoint = {  'pretrained_model_name': model_name,
                    'model.classifier': model.classifier,
                    'model.class_to_idx': model.class_to_idx,
                    'model.next_epoch': model.next_epoch,
                    'model.state_dict': model.state_dict(),
                    'optimizer_name': optimizer_name,
                    'optimizer.lr': optimizer.lr,
                    'optimizer.state_dict': optimizer.state_dict(),
                    'criterion_name': criterion_name
    }

    torch.save(checkpoint, checkpoint_name)
    
def load_checkpoint(checkpoint_filepath, device):
    '''
    Regenerate model, optimizer and criterion from checkpoint file
    '''
    checkpoint = torch.load(checkpoint_filepath)
    model_constr = getattr(models, checkpoint['pretrained_model_name'])
    model = model_constr(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = checkpoint['model.classifier']
    model.to(device)
    model.class_to_idx = checkpoint['model.class_to_idx']
    model.idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    model.next_epoch = checkpoint['model.next_epoch']
    model.load_state_dict(checkpoint['model.state_dict'])
    
    optimizer_constr = getattr(optim, checkpoint['optimizer_name'])
    optimizer = optimizer_constr(model.classifier.parameters(), lr=checkpoint['optimizer.lr'])
    optimizer.load_state_dict(checkpoint['optimizer.state_dict'])
    
    criterion_constr = getattr(nn, checkpoint['criterion_name'])
    criterion = criterion_constr()
    
    return model, optimizer, criterion