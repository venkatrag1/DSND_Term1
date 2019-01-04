#!/usr/bin/env python
import argparse
import os, sys
from collections import OrderedDict
from workspace_utils import active_session
from utils import get_image_data, save_checkpoint

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import models

DEVICE = torch.device('cpu')

def parse_args():
    parser = argparse.ArgumentParser(description="train deep neural network with images in folder and generate trained_model checkpoint")
    parser.add_argument('data_dir', help='directory containing three subdirectories named train, test and valid, '
                        'each containing labeled images conforming to ImageFolder format')
    parser.add_argument('--save_dir', default='.', help='directory to save trained_model.pth')
    parser.add_argument('--arch', default='vgg16', help='pretrained network to use')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')   
    parser.add_argument('--hidden_units', nargs=2, type=int, default=[4096, 1024], help='number of hidden units in hidden layer 1 and 2') 
    parser.add_argument('--epochs', type=int, default=10, help='num epochs to train for') 
    parser.add_argument('--gpu', action='store_true', default=False, help='use GPU to train') 
    return parser.parse_args()

def get_training_base(arch, hidden_units, out_features, learning_rate):
    # Ref- https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    model_constr = getattr(models, arch)
    model = model_constr(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    if "vgg" in args.arch or "alexnet" in args.arch:
        in_features = model.classifier[0].in_features
    else:
        print("Currently only vgg and alexnet architectures are supported")
        sys.exit(0)
    model.classifier = nn.Sequential(OrderedDict([
                                 ('fc1', nn.Linear(in_features, hidden_units[0])),
                                 ('fc1_relu', nn.ReLU()),
                                 ('fc1_dropout', nn.Dropout(0.5)),
                                 ('fc2_lin', nn.Linear(hidden_units[0], hidden_units[1])),
                                 ('fc2_relu', nn.ReLU()),
                                 ('fc2_dropout', nn.Dropout(0.5)),
                                 ('out_lin', nn.Linear(hidden_units[1], out_features)),
                                 ('out_logsoftmax', nn.LogSoftmax(dim=1))
                                  ]))
    model.next_epoch = 1
    criterion = nn.NLLLoss()
    model.to(DEVICE);
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    optimizer.lr = learning_rate                 
    return model, criterion, optimizer

def evaluate_model(model, dataloader, criterion):
    eval_loss = 0
    eval_accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            eval_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            eval_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    return eval_loss/ len(dataloader), eval_accuracy / len(dataloader)

def train_model(model, train_dataloader, criterion, 
                optimizer, validation_dataloader,
                last_epoch):
    steps = 0
    running_loss = 0
    PRINT_EVERY = 20
    first_epoch = model.next_epoch
    with active_session():
        for epoch in range(first_epoch, last_epoch+1):
            for inputs, labels in train_dataloader:
                steps += 1
                # Move input and label tensors to the default device
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                optimizer.zero_grad()

                logps = model.forward(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % PRINT_EVERY == 0:
                    validation_loss, validation_accuracy = evaluate_model(
                                                            model, 
                                                            validation_dataloader, 
                                                            criterion)

                    print(f"Epoch {epoch}/{last_epoch}.. "
                          f"Train loss: {running_loss/PRINT_EVERY:.3f}.. "
                          f"Validation loss: {validation_loss:.3f}.. "
                          f"Validation accuracy: {validation_accuracy:.3f}")
                    running_loss = 0
                    model.train()
    model.next_epoch = last_epoch + 1
    return model, optimizer
        
def main(args):
    global DEVICE
    if args.gpu:
        if torch.cuda.is_available():
            DEVICE = torch.device('cuda')
        else:
            print("CUDA is not available, reverting to CPU")
    
    image_datasets, dataloaders = get_image_data(args.data_dir)
    out_features = len(os.listdir(args.data_dir + '/train'))
    model, criterion, optimizer = get_training_base(args.arch, args.hidden_units,
                                         out_features, args.learning_rate)
    model.class_to_idx = image_datasets['train'].class_to_idx
    model.idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    model, optimizer = train_model(model, dataloaders['train'], criterion, 
                                  optimizer, dataloaders['valid'], args.epochs)
    test_loss, test_accuracy = evaluate_model(
                                model, 
                                dataloaders['test'],
                                criterion)
    print(f"Test loss: {test_loss:.3f}.. "
          f"Test Accuracy: {test_accuracy:.3f}")
    save_checkpoint(model, optimizer, args.arch, "Adam", "NLLLoss", 
                    checkpoint_name=os.path.join(args.save_dir, 'trained_model.pth'))
    
if __name__ == "__main__":
    args = parse_args()
    main(args)

