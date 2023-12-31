"""
The only useful resource I found on QAT training was at:
https://support.xilinx.com/s/article/How-to-make-a-Quantization-Aware-Training-QAT-with-a-model-developed-in-a-Pytorch-framework?language=en_US

Here is the script it provides:
"""

'''
 Copyright 2020 Xilinx Inc.
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
     http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
'''

'''
Simple PyTorch MNIST example - training & testing
Author: Mark Harvey, Xilinx 
'''

'''
QAT update for AR000033688 by Giovanni Guasti, Xilinx inc
run with# run QAT training
python -u train_QAT_CPU.py -d ${BUILD} --epochs 1 2>&1 | tee ${LOG}/train_QAT.log
python -u train_QAT_GPU.py -d ${BUILD} --epochs 1 2>&1 | tee ${LOG}/train_QAT.log
'''

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import argparse
import sys
import os
import shutil

from common import *
from pytorch_nndct.apis import torch_quantizer, dump_xmodel

DIVIDER = '-----------------------------------------'

torchvision.datasets.MNIST.resources = [
    ('https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz', 'f68b3c2dcbeaaa9fbdd348bbdeb94873'),
    ('https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz', 'd53e105ee54ea40749a09fcbcd1e9432'),
    ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz', '9fb629c4189551a2d022fa330f9573f3'),
    ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz', 'ec29112dd5afa0611ce80d1b7f02629c')
]

def train_test(build_dir, batchsize, learnrate, epochs):

    dset_dir = build_dir + '/dataset'
    float_model = build_dir + '/float_model'

    '''
    # use GPU if available   
    if (torch.cuda.device_count() > 0):
        print('You have',torch.cuda.device_count(),'CUDA devices available')
        for i in range(torch.cuda.device_count()):
            print(' Device', str(i),': ',torch.cuda.get_device_name(i))
        print('Selecting device 0..')
        device = torch.device('cuda:0')
    else:
        print('No CUDA devices available..selecting CPU')
        device = torch.device('cpu')
    '''
    print('This example runs Train and xmodel generation all on CPU')
    device = torch.device('cpu')

    model = CNN_QAT().to(device)
    
    #rand_in = torch.randn([batchsize, 1, 28, 28], dtype=torch.float32)
    rand_in = torch.randn([batchsize, 1, 28, 28]).to(device)
    
    # Create a quantizer
    from pytorch_nndct import QatProcessor
    qat_processor = QatProcessor(model, rand_in, bitwidth=8, device= device)


    quantized_model = qat_processor.trainable_model()
    
    optimizer = torch.optim.Adam(
        quantized_model.parameters(), 
        lr=learnrate, 
        weight_decay=0)

    #image datasets
    train_dataset = torchvision.datasets.MNIST(dset_dir, 
                                               train=True, 
                                               download=True,
                                               transform=train_transform)
    test_dataset = torchvision.datasets.MNIST(dset_dir,
                                              train=False, 
                                              download=True,
                                              transform=test_transform)
    #loaders
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batchsize, 
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batchsize, 
                                              shuffle=False)

    # training with test after each epoch
    for epoch in range(1, epochs + 1):
        train(quantized_model, device, train_loader, optimizer, epoch)
        test(quantized_model, device, test_loader)

    # --------------------------------------------------------
    # get deployable model and test it
    output_dir = 'qat_result'
    batchsize = 1
    test_loader = torch.utils.data.DataLoader(test_dataset,
        batch_size=batchsize, 
        shuffle=False)

    deployable_model = qat_processor.to_deployable(quantized_model, output_dir)
    test(deployable_model, device, test_loader)

    # --------------------------------------------------------
    # Export xmodel from deployable model
    # use CPU mode
   
    # Must forward deployable model at least 1 iteration with batchsize=1
    for images, _ in test_loader:
        deployable_model(images.to(device))
    
    qat_processor.export_xmodel(output_dir)

    return


def run_main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--build_dir',   type=str,  default='build',       help='Path to build folder. Default is build')
    ap.add_argument('-b', '--batchsize',   type=int,  default=100,           help='Training batchsize. Must be an integer. Default is 100')
    ap.add_argument('-e', '--epochs',      type=int,  default=3,             help='Number of training epochs. Must be an integer. Default is 3')
    ap.add_argument('-lr','--learnrate',   type=float,default=0.001,         help='Optimizer learning rate. Must be floating-point value. Default is 0.001')
    args = ap.parse_args()

    print('\n'+DIVIDER)
    print('PyTorch version : ',torch.__version__)
    print(sys.version)
    print(DIVIDER)
    print(' Command line options:')
    print ('--build_dir    : ',args.build_dir)
    print ('--batchsize    : ',args.batchsize)
    print ('--learnrate    : ',args.learnrate)
    print ('--epochs       : ',args.epochs)
    print(DIVIDER)

    train_test(args.build_dir, args.batchsize, args.learnrate, args.epochs)

    return



if __name__ == '__main__':
    run_main()