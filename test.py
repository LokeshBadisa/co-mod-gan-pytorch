"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import pdb
import cv2
import os
from collections import OrderedDict
import json
from tqdm import tqdm
import numpy as np
import torch
import data
from options.test_options import TestOptions
#from models.pix2pix_model import Pix2PixModel
import models
from util.util import inverse_transform
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from torch import nn

opt = TestOptions().parse()

dataloader = data.create_dataloader(opt,'val')

model = models.create_model(opt)
model.eval()
Path('./results/').mkdir(parents=True, exist_ok=True)
dilation1 = nn.MaxPool2d(5, stride=1, padding=2)

for i, data_i in enumerate(tqdm(dataloader)):
    if i * opt.batchSize >= opt.how_many:
        break
    with torch.no_grad():
        data_i['mask'] = dilation1(data_i['mask'])
        generated,_ = model(data_i, mode='inference')
    # generated = torch.clamp(generated, -1, 1)
    generated = inverse_transform(generated)*255
    generated = generated.cpu().numpy().astype(np.uint8)
    # print(data_i['path'])
    img_path = data_i['path']
    for b in range(generated.shape[0]):
        pred_im = generated[b].transpose((1,2,0))
        # print('process image... %s' % img_path[b])
        # cv2.imwrite('./results/'+img_path[b], pred_im[:,:,::-1])
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(inverse_transform(data_i['image'][b]).cpu().numpy().transpose((1,2,0)))
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.imshow(pred_im)
        plt.axis('off')
        plt.savefig('./results/'+img_path[b])

