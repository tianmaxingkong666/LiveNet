from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from align import detect_face
import argparse
import time
import yaml
import torch
import torch.nn as nn
import torch.nn.parallel
import torchvision.transforms as transforms
from utils.profile import count_params
import os
from torch.autograd.variable import Variable
import models
from PIL import Image
import numpy as np
import cv2
import imutils
import torch.nn.functional as F
import torchvision
from imutils.video import VideoStream
from scipy import misc

print('Creating networks and loading parameters')
    
with tf.Graph().as_default():
    sess = tf.Session(config=tf.ConfigProto())
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

MODEL_PATH = 'checkpoints\\mobilenetv2_bs32\\_20_best.pth.tar'
MODEL_ARCH = 'moilenetv2'
model = models.__dict__[MODEL_ARCH]()
model = torch.nn.DataParallel(model)
print("=> loading checkpoint '{}'".format(MODEL_PATH))
checkpoint = torch.load(MODEL_PATH,map_location='cpu')
model.load_state_dict(checkpoint['state_dict'])
model.eval()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
img_size = 224
ratio = 224.0 / float(img_size)
transform = transforms.Compose([
        transforms.Resize(int(256 * ratio)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
        ])

vs = VideoStream(src=0).start()
time.sleep(2.0)

minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor
margin = 22
image_size = 224


while True:
    src_frame = vs.read()
    print(src_frame.shape)
    r,g,b = cv2.split(src_frame)
    frame = cv2.merge([b,g,r])
    img_size = np.asarray(frame.shape)[0:2]
    bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
    for i in range(0,bounding_boxes.shape[0]):
        confidence = bounding_boxes[i][4]
        print(confidence)
        if confidence > 0.8:
            det = bounding_boxes[i,0:4]
            det = np.squeeze(det)
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0]-margin/2, 0)
            bb[1] = np.maximum(det[1]-margin/2, 0)
            bb[2] = np.minimum(det[2]+margin/2, img_size[1])
            bb[3] = np.minimum(det[3]+margin/2, img_size[0])
            startY = bb[1]
            endY = bb[3]
            startX = bb[0]
            endX = bb[2]
            #cropped = frame[bb[1]:bb[3],bb[0]:bb[2],:]
            face = frame[startY:endY, startX:endX]
            face = misc.imresize(face, (image_size, image_size), interp='bilinear')
            image = Image.fromarray(face)
            image = transform(image)
            print(image.shape)
            image = image.view(1, *image.size())
            print(image.shape) 
            #image = image.cuda()
            image = Variable(image, volatile=True)
            preds = model(image)
            print(preds)
            _, pred = preds.topk(1, 1, True, True)
            ret = pred.item()
            print(ret)
            preds = F.softmax(preds,dim=1) # 按行SoftMax,行和为1
            preds = preds.squeeze(0)
            print(preds)
            if ret == 1:
                score = preds[1].detach().numpy()
                #print('##############')
                #print(score)
                label = 'real: '+ str(score)
            else:
                score = preds[0].detach().numpy()
                label = 'fake: ' + str(score)
            print(label)
            
            # draw the label and bounding box on the frame
            cv2.putText(src_frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.rectangle(src_frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
    # show the output frame and wait for a key press
    cv2.imshow("Frame", src_frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()