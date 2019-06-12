import numpy as np
from tqdm import tqdm
import torch

from modeling.deeplab import *
from modeling.sync_batchnorm.replicate import patch_replication_callback
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from PIL import Image
import time
import cv2
import torch.nn.functional as F
from dataloaders import utils
from res18test.resnet_dilated import *
from doc.deeplab_resnet import *

#import pydensecrf.densecrf as dcrf
import os
model = DeepLab(num_classes=2,
                backbone='mobilenet',
                output_stride=32,
                sync_bn=False,
                freeze_bn=False)

model = torch.nn.DataParallel(model, device_ids=[0,1])
patch_replication_callback(model)
model = model.cuda()
checkpoint = torch.load("/home/xupeihan/deeplab/run/vocdetection/mb_finAL/experiment_2/checkpoint.pth.tar")
model.module.load_state_dict(checkpoint['state_dict'])
#model.load_state_dict(checkpoint['state_dict'])

model.eval()
mean=(0.485, 0.456, 0.406) 
std=(0.229, 0.224, 0.225)

time1 = time.time()

PATH = '/home/xupeihan/deeplab/img/'
#PATH = '/mnt/disk2/xupeihan/seg_code/res-101-1-25mix/img/'
imageTest = os.listdir(PATH)
 
for imageName in imageTest:

    image = Image.open(PATH+imageName)
    h,w = image.size
    image = image.resize((513,513),Image.NEAREST)
    image_array = np.array(image).astype(np.float32)
    image_array /= 255.0
    image_array -= mean
    image_array /= std
    #print(image_array.shape)
    image_array = image_array.transpose((2,0,1))
    img = torch.from_numpy(image_array).float()
    #img = img.cuda()
    img = img.unsqueeze(0)
    #print(img.shape)
    with torch.no_grad():
        output = model(img)
    output = output[0].data.cpu().numpy()  
    
    output = np.argmax(output,axis=1)
    #print(output.shape)
    output = output.squeeze(0)
    print(sum(sum(output)))
    #print(output.shape)
    output[output==1] = 255
    #output = utils.decode_segmap(output,dataset='pascal')
    #print(output.shape)
    #output *= 255.0

    pred = Image.fromarray(np.uint8(output))
    pred = pred.resize((h,w),Image.NEAREST)
    pred.save('./img_result/'+imageName)
