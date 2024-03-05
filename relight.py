import sys
sys.path.append('model')
sys.path.append('utils')

from utils_SH import *

# other modules
import os
import numpy as np
import argparse

from torch.autograd import Variable
from torchvision.utils import make_grid
import torch
import time, math
import cv2

from pathlib import Path

from defineHourglass_1024_gray_skip_matchFeature import *


parser = argparse.ArgumentParser()
parser.add_argument("input_path")
parser.add_argument("output_path")
args = parser.parse_args()

# # ---------------- create normal for rendering half sphere ------
# img_size = 256
# x = np.linspace(-1, 1, img_size)
# z = np.linspace(1, -1, img_size)
# x, z = np.meshgrid(x, z)

# mag = np.sqrt(x**2 + z**2)
# valid = mag <=1
# y = -np.sqrt(1 - (x*valid)**2 - (z*valid)**2)
# x = x * valid
# y = y * valid
# z = z * valid
# normal = np.concatenate((x[...,None], y[...,None], z[...,None]), axis=2)
# normal = np.reshape(normal, (-1, 3))
# #-----------------------------------------------------------------

modelFolder = 'trained_model/'

# load model
from defineHourglass_512_gray_skip import *
my_network = HourglassNet()
my_network.load_state_dict(torch.load(os.path.join(modelFolder, 'trained_model_03.t7')))
my_network.cuda()
my_network.train(False)


lights = [
    ["l0_0", "l0_1", "l0_2"],
    ["l1_0", "l1_1", "l1_2"],
    ["l2_0", "l2_1", "l2_2"]
]

def relight_grid(input_path, output_path):
    output_path = Path(output_path)
    input_path  = Path(input_path)
    lightFolder = 'data/envmaps/sh'

    img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    img_sml = cv2.resize(img, (512, 512))
    alpha = img_sml[:,:,3]

    Lab = cv2.cvtColor(img_sml[:,:,0:3], cv2.COLOR_BGR2LAB)

    inputL = Lab[:,:,0]
    inputL = inputL.astype(np.float32)/255.0
    inputL = inputL.transpose((0,1))
    inputL = inputL[None,None,...]
    inputL = Variable(torch.from_numpy(inputL).cuda())

    shs = []

    for i, name in enumerate([name for ls in lights for name in ls]):
            path = Path(lightFolder, f"{name}.txt")
            sh = np.loadtxt(path)
            sh = sh[0:9]
            sh = sh * 0.5
            sh = np.squeeze(sh)
            shs.append((name,sh))

    results = {}
    for i, (name, sh) in enumerate(shs):

        sh *= 0.01

        # rendering half-sphere
        # shading = get_shading(normal, sh)
        # value = np.percentile(shading, 95)
        # ind = shading > value
        # shading[ind] = value
        # shading = (shading - np.min(shading))/(np.max(shading) - np.min(shading))
        # shading = (shading *255.0).astype(np.uint8)
        # shading = np.reshape(shading, (256, 256))
        # shading = shading * valid
        # cv2.imwrite(os.path.join(saveFolder, \
        #         'light_{:02d}.png'.format(i)), shading)

        #----------------------------------------------
        #  rendering images using the network
        #----------------------------------------------
        sh = np.reshape(sh, (1,9,1,1)).astype(np.float32)
        sh = Variable(torch.from_numpy(sh).cuda())
        outputImg, outputSH  = my_network(inputL, sh, 0)
        outputImg = outputImg[0].cpu().data.numpy()
        outputImg = outputImg.transpose((1,2,0))
        outputImg = np.squeeze(outputImg)
        outputImg = (outputImg*255.0).astype(np.uint8)
        Lab[:,:,0] = outputImg
        resultLab = cv2.cvtColor(Lab, cv2.COLOR_LAB2BGR)
        result = cv2.cvtColor(resultLab, cv2.COLOR_BGR2BGRA)
        result[:,:,3] = alpha

        result = cv2.resize(result, (img.shape[1], img.shape[0]))
        image = 127 + (img[:,:,0:3] - result[:,:,0:3])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        image[:,:,3] = img[:,:,3]
        results[name] = image

        # output_file_path = Path(output_path, f"{input_path.stem}_light_{name}.png")
        # cv2.imwrite(str(output_file_path.resolve()), img )

        # output_file_path = Path(output_path, f"_{input_path.stem}_light_{name}.png")
        # cv2.imwrite(str(output_file_path.resolve()), image )



    rows = []
    for row in lights:
         row_images = [cv2.resize(results[name], (0,0), fx=1.0/3.0, fy=1.0/3.0) for name in row]
         rows.append(cv2.hconcat(row_images))
         
    result = cv2.vconcat(rows)
    result = cv2.resize(result, (img.shape[1], img.shape[0]))
    cv2.imwrite(str(output_path.resolve()), result)

if __name__ == "__main__":
    input_path  = args.input_path
    output_path = args.output_path
    relight_grid(input_path, output_path)