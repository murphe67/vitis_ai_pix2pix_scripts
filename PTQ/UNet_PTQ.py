"""
This was the final PTQ script for the U-Net- 
I couldn't be bothered to figure out how to change the U-Net's super complicated parser, so instead of 
having calib vs. export be done via argument, I just changed the file manually based on what I needed.

It's currently in calib mode.

Again one of the main difficulties is figuring out how to load your .pth file into the quantizer.
"""

from __future__ import print_function
import argparse
import os

import time

import torch
import torchvision.transforms as transforms
from pytorch_nndct.apis import torch_quantizer, dump_xmodel

from utils import is_image_file, load_img, save_img

from models import create_model

from options.test_options import TestOptions

# original settings
# parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
# parser.add_argument('--dataset', required=True, help='facades')
# parser.add_argument('--direction', type=str, default='b2a', help='a2b or b2a')
# parser.add_argument('--nepochs', type=int, default=200, help='saved model of which epochs')
# parser.add_argument('--cuda', action='store_true', help='use cuda')
# opt = parser.parse_args()
# print(opt)

# Testing settings
opt = TestOptions().parse()  
# hard-code some parameters for test
opt.num_threads = 0   # test code only supports num_threads = 0
opt.batch_size = 1    # test code only supports batch_size = 1
opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
#dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
model = create_model(opt)  

device = torch.device("cpu")

model.load_networks("latest")
network = model.netG


net_g = network.to(device)



quant_model = "quant/"

rand_in = torch.randn([1, 3, 256, 256])
#change based on calib or test
quantizer = torch_quantizer('calib', net_g, (rand_in), output_dir=quant_model) 
quantized_model = quantizer.quant_model



if opt.direction == "a2b":
    image_dir = "dataset/facades/test/a/"
else:
    image_dir = "dataset/facades/test/b/"

image_dir = "dataset/matting_vitis_pix2pix/"

image_filenames = [x for x in os.listdir(image_dir) if is_image_file(x)]

transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

transform = transforms.Compose(transform_list)

#for image_name in image_filenames:
for i in range(3):
    image_name = image_filenames[i]
    img = load_img(image_dir + image_name)
    img = transform(img)
    t1 = time.time()
    input = img.unsqueeze(0).to(device)
    out = quantized_model(input)
    out_img = out.detach().squeeze(0).cpu()
    t2 = time.time()

    print("time taken: " + str(t2 - t1))
    

    if not os.path.exists(os.path.join("result", "facades")):
        os.makedirs(os.path.join("result", "facades"))
    save_img(out_img, "result/facades/{}".format("unet_"+ image_name))

#change based on calib or test
quantizer.export_quant_config()