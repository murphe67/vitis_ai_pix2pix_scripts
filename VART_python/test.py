from __future__ import print_function
import argparse
import os

import torch
import torchvision.transforms as transforms

from statistics import mean

from ctypes import *
from typing import List
import cv2
import numpy as np
import vart
import os
import pathlib
import xir
import threading
import time
import sys
import argparse

from PIL import Image

from scipy.io import savemat

from utils import is_image_file, load_img, save_img

_divider = '-------------------------------'


def preprocess_fn(image_path, input_scale, transform, image_number):
    img = load_img(image_path)
    img = transform(img).float().numpy()

    img = img * input_scale

    img = img.astype(np.int8)
    img = np.transpose(img, (1, 2, 0))

    return img



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

def runDPU(threadID,start,dpu,img):
    for i in range(numberReruns):
        '''get tensor'''
        inputTensors = dpu.get_input_tensors()
        outputTensors = dpu.get_output_tensors()
        input_ndim = tuple(inputTensors[0].dims)
        output_ndim = tuple(outputTensors[0].dims)

        output_fixpos = outputTensors[0].get_attr("fix_point")
        output_scale = 1 / (2**output_fixpos)

        batchSize = input_ndim[0]
        n_of_images = len(img)
        count = 0
        write_index = start
        ids=[]
        ids_max = 10
        outputData = []

        for i in range(ids_max):
            outputData.append([np.empty(output_ndim, dtype=np.int8, order="C")])
        while count < n_of_images:
            if (count+batchSize<=n_of_images):
                runSize = batchSize
            else:
                runSize=n_of_images-count

            '''prepare batch input/output '''
            inputData = []
            inputData = [np.empty(input_ndim, dtype=np.int8, order="C")]

            '''init input image to input buffer '''
            times = []
            for j in range(runSize):
                imageRun = inputData[0]
                imageRun[j, ...] = img[(count + j) % n_of_images].reshape(input_ndim[1:])
            '''run with batch '''
            job_id = dpu.execute_async(inputData,outputData[len(ids)])
            global powerID
            powerID += 1
            
            ids.append((job_id,runSize,start+count, time.time()))
            count = count + runSize 
            if count<n_of_images:
                if len(ids) < ids_max-1:
                    continue
            for index in range(len(ids)):
                dpu.wait(ids[index][0])
                

                global latency
                latency += time.time() - ids[index][3]
                write_index = ids[index][2]
                '''store output vectors '''
                for j in range(ids[index][1]):
                    out_q[write_index] = outputData[index][0][j] * output_scale
                    write_index += 1
            ids=[]


def main():
    image_dir = "/workspace/dataset/facades/test/a/"

    image_filenames = [x for x in os.listdir(image_dir) if is_image_file(x)]
    


    runTotal = len(image_filenames)
    #runTotal = 200
    global numberReruns
    numberReruns = 1


    global realRunTotal
    realRunTotal = runTotal * numberReruns
    #runTotal = 100

    global out_q
    out_q = [None] * runTotal

    global threads
    threads = 2

    global latency
    latency = 0

    global powerID
    powerID = 0
    
    g = xir.Graph.deserialize("/workspace/pic2pic_u50_qat_label.xmodel")
    subgraphs = get_child_subgraph_dpu(g)

    all_dpu_runners = []
    for i in range(threads):
        all_dpu_runners.append(vart.Runner.create_runner(subgraphs[0], "run"))

    # input scaling
    input_fixpos = all_dpu_runners[0].get_input_tensors()[0].get_attr("fix_point")
    input_scale = 2**input_fixpos

    transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    transform = transforms.Compose(transform_list)


    ''' preprocess images '''
    print (_divider)
    print('Pre-processing',runTotal,'images...')
    img = []
    for i in range(runTotal):
        image_name = image_filenames[i]
        img.append(preprocess_fn(image_dir + image_name, input_scale, transform, i))

    '''run threads '''
    print (_divider)
    print('Starting',threads,'threads...')
    threadAll = []
    start=0
    for i in range(threads):
        if (i==threads-1):
            end = len(img)
        else:
            end = start+(len(img)//threads)
        in_q = img[start:end]
        t1 = threading.Thread(target=runDPU, args=(i,start,all_dpu_runners[i], in_q))
        threadAll.append(t1)
        start=end


    time1 = time.time()
    for x in threadAll:
        x.start()
    for x in threadAll:
        x.join()
    time2 = time.time()
    timetotal = time2 - time1

    fps = float(realRunTotal / timetotal)
    print (_divider)
    print("Throughput=%.2f fps, total frames = %.0f, time=%.4f seconds, avg latency = %.4f" %(fps, realRunTotal, timetotal, latency / realRunTotal))

    transform = transforms.ToTensor()
    for i in range(runTotal):
        save_img(transform(out_q[i]), "result/facades_label/UNet_FPGA_QAT_{0:0>4}.png".format(i))

if __name__ == '__main__':
  main()

latency = 0
powerID = 0