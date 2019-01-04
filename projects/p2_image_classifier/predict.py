#!/usr/bin/env python
import argparse
import os, sys, json
from utils import process_image, load_checkpoint

import numpy as np
import torch
from torch.autograd import Variable

from PIL import Image

DEVICE = torch.device('cpu')

def parse_args():
    '''
    Parse command line args
    '''
    parser = argparse.ArgumentParser(description="predict image class (and class name if mapping is provided)"
                                     "for a given image from a pretrained network loaded from checkpoint")
    parser.add_argument('path_to_image', help='path to the image to use for prediction')
    parser.add_argument('checkpoint', help='path to checkpoint for the trained model')
    parser.add_argument('--top_k', type=int, default=1, help='prints out top_k predictions')
    parser.add_argument('--category_names', default=None, help='path to category to name mapping json file')
    parser.add_argument('--gpu', action='store_true', default=False, help='use GPU to train') 
    return parser.parse_args()

def predict(image_path, model, topk=5):
    ''' 
    Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    # http://blog.outcome.io/pytorch-quick-start-classifying-an-image/ - Add dimension to simulate batch when 
    # converting to tensor manually and then convert to Autograd variable
    model.eval()
    with torch.no_grad():
        # Load image as PIL, process it and generate FloatTensor 
        inputs = torch.from_numpy(process_image(Image.open(image_path))).type(torch.FloatTensor)
        # Simulate batch
        inputs = inputs.unsqueeze_(0)
        # Convert to autograd
        inputs = Variable(inputs)
        inputs = inputs.to(DEVICE)
        # Forward pass and get prob
        logps = model.forward(inputs)
        ps = torch.exp(logps)
        top_p, top_idx = ps.topk(topk, dim=1)
        # Convert to np array and then squeeze first dimension manually (to prevent squeezing top_k=1 case)
        top_prob = np.array(top_p)[0]
        top_class = np.array([model.idx_to_class[idx] for idx in np.array(top_idx)[0]])
        return top_prob, top_class
    
def main(args):
    global DEVICE
    if args.gpu:
        if torch.cuda.is_available():
            DEVICE = torch.device('cuda')
        else:
            print("CUDA is not available, reverting to CPU")
    # Get model
    model, optimizer, criterion = load_checkpoint(args.checkpoint, DEVICE)
    # Make prediction
    top_prob, top_class = predict(args.path_to_image, model, topk=args.top_k)
    output_header = "(CLASS): PROBABILITY"
    output_rows = [f"({klass}): {prob}" for klass, prob in zip(top_class, top_prob)]
    # Add category name if available
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
            output_rows = [f"{cat_to_name[top_class[i]]} " + row for i, row in enumerate(output_rows)] 
            output_header = f"CLASS_NAME " + output_header
    print(output_header)
    print('--'*30)
    [print(row) for row in output_rows]
      
if __name__ == "__main__":
    args = parse_args()
    main(args)


