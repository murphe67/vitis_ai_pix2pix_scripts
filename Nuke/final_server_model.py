# Copyright (c) 2019 Foundry.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

from ..baseModel import BaseModel

import cv2
import numpy as np

from typing import List

import xir
import vart

import os

from PIL import Image
import torchvision.transforms as transforms
import threading
import time
import sys
import argparse

#from ..common.util import linear_to_srgb, srgb_to_linear

import message_pb2

class Model(BaseModel):
    def __init__(self):
        super(Model, self).__init__()
        self.name = 'Pic2Pic Facades'

        # Define inputs/outputs
        self.inputs = {'input': 3}
        self.outputs = {'output': 3}


        self.threads = 2
        
        g = xir.Graph.deserialize("pic2pic_u50_unet.xmodel")
        subgraphs = get_child_subgraph_dpu(g)

        # self.dpu = vart.Runner.create_runner(subgraphs[0], "run")
        self.all_dpu_runners = []
        for i in range(self.threads):
            self.all_dpu_runners.append(vart.Runner.create_runner(subgraphs[0], "run"))

        # input scaling
        input_fixpos = self.all_dpu_runners[0].get_input_tensors()[0].get_attr("fix_point")
        self.input_scale = 2**input_fixpos

        inputTensors = self.all_dpu_runners[0].get_input_tensors()
        outputTensors = self.all_dpu_runners[0].get_output_tensors()
        self.input_ndim = tuple(inputTensors[0].dims)
        self.output_ndim = tuple(outputTensors[0].dims)

        # we can avoid output scaling if use argmax instead of softmax
        output_fixpos = outputTensors[0].get_attr("fix_point")
        self.output_scale = 1 / (2**output_fixpos)

    def inference(self, image_list):
        inputImage = image_list[0]
        inHeight, inWidth, inChannels = inputImage.shape
        widthResize = 256 * round(inWidth / 256)
        heightResize = 256 * round(inHeight / 256)
        inputImage = cv2.resize(inputImage, (widthResize, heightResize), interpolation = cv2.INTER_AREA)
        inputImage = linear_to_srgb(inputImage)
        inputImage = (inputImage * 255).astype(np.uint8)
        
        #inputImage = cv2.resize(inputImage, (256, 256), interpolation = cv2.INTER_AREA)

        # print(inputImage.shape)    

        unpackedImages = []
        for i in range(0, widthResize, 256):
            for j in range(0, heightResize, 256):
                unpackedImages.append(inputImage[j:j+256, i:i+256, :])


        self.mainOther(unpackedImages)

        imageIndex = 0
        outputImage = np.empty((heightResize, widthResize, 3), order="C")

        for i in range(0, widthResize, 256):
            for j in range(0, heightResize, 256):
                outputImage[j:j+256, i:i+256, :] = out_q[imageIndex]
                imageIndex += 1

        transform = transforms.ToTensor()

        image_numpy = transform(outputImage).float().numpy()
        image_numpy = (image_numpy + 1) / 2.0 * 255.0
        image_numpy = image_numpy.clip(0, 255)
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
        image_numpy = cv2.resize(image_numpy, (inWidth, inHeight), interpolation = cv2.INTER_AREA)
        image_numpy = image_numpy.astype(np.uint8)


        out = image_numpy.astype(np.float32) / 255.
        out = srgb_to_linear(out)
        return [out]


    def mainOther(self, inputImages):
        runTotal = len(inputImages)

        global out_q
        out_q = [None] * runTotal

        global threads
        threads = 1
        
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        transform = transforms.Compose(transform_list)

        ''' preprocess images '''
        print (_divider)
        print('Pre-processing',runTotal,'images...')
        img = []
        for i in range(runTotal):
            img.append(preprocess_fn(inputImages[i], self.input_scale, transform))

        # '''run threads '''
        # print (_divider)
        # print('Starting execution...')

        # time1 = time.time()
        # self.runDPU(0, 0, img)
        # time2 = time.time()
        # timetotal = time2 - time1

        '''run threads '''
        print (_divider)
        print('Starting',self.threads,'threads...')
        threadAll = []
        start=0
        for i in range(self.threads):
            if (i==self.threads-1):
                end = len(img)
            else:
                end = start+(len(img)//self.threads)
            in_q = img[start:end]
            t1 = threading.Thread(target=self.runDPU, args=(i,start,self.all_dpu_runners[i], in_q))
            threadAll.append(t1)
            start=end


        time1 = time.time()
        for x in threadAll:
            x.start()
        for x in threadAll:
            x.join()
        time2 = time.time()
        timetotal = time2 - time1

        fps = float(runTotal / timetotal)
        print (_divider)
        print("Throughput=%.2f fps, total frames = %.0f, time=%.4f seconds" %(fps, runTotal, timetotal))

    def runDPU(self,id,start, dpu, img):
        batchSize = self.input_ndim[0]
        n_of_images = len(img)
        count = 0
        write_index = start
        ids=[]
        ids_max = 10
        outputData = []
        
        for i in range(ids_max):
            outputData.append([np.empty(self.output_ndim, dtype=np.int8, order="C")])
        while count < n_of_images:
            if (count+batchSize<=n_of_images):
                runSize = batchSize
            else:
                runSize=n_of_images-count

            '''prepare batch input/output '''
            inputData = []
            inputData = [np.empty(self.input_ndim, dtype=np.int8, order="C")]

            '''init input image to input buffer '''
            for j in range(runSize):
                imageRun = inputData[0]
                imageRun[j, ...] = img[(count + j) % n_of_images].reshape(self.input_ndim[1:])
            '''run with batch '''
            job_id = dpu.execute_async(inputData,outputData[len(ids)])

            ids.append((job_id,runSize,start+count))
            count = count + runSize 
            if count<n_of_images:
                if len(ids) < ids_max-1:
                    continue
            for index in range(len(ids)):
                dpu.wait(ids[index][0])
                #os.system("xbutil query");
                write_index = ids[index][2]
                '''store output vectors '''
                for j in range(ids[index][1]):
                    out_q[write_index] = outputData[index][0][j] * self.output_scale
                    write_index += 1
            ids=[]

_divider = '-------------------------------'

def srgb_to_linear(x):
    """Transform the image from sRGB to linear"""
    a = 0.055
    x = np.clip(x, 0, 1)
    mask = x < 0.04045
    x[mask] /= 12.92
    x[mask!=True] = np.exp(2.4 * (np.log(x[mask!=True] + a) - np.log(1 + a)))
    return x

def linear_to_srgb(x):
    """Transform the image from linear to sRGB"""
    a = 0.055
    x = np.clip(x, 0, 1)
    mask = x <= 0.0031308
    x[mask] *= 12.92
    x[mask!=True] = np.exp(np.log(1 + a) + (1/2.4) * np.log(x[mask!=True])) - a
    return x

def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (root_subgraph is not None), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]


def preprocess_fn(img, input_scale, transform):
    img = transform(img).float().numpy()
    img = img * input_scale
    img = img.astype(np.int8)
    img = np.transpose(img, (1, 2, 0))

    return img


