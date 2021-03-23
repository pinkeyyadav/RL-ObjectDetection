import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable
import random
import matplotlib.pyplot as plt
from __init__ import *
from annotate import *
from features import *
from reinforcement import *
import logging
import time
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import pandas as pd



def draw_bounding_box_1(annotation, img):
    new_img = Image.fromarray(img)
    draw = ImageDraw.Draw(new_img)
    length = len(annotation)
    annotation = torch.matrix(annotation)
    for i in range(length):
        x_min = int(annotation[i, 1])
        x_max = int(annotation[i, 2])
        y_min = int(annotation[i, 3])
        y_max = int(annotation[i, 4])
        draw.line(((x_min, y_min), (x_max, y_min)), fill="red", width=3)
        draw.line(((x_min, y_min), (x_min, y_max)), fill="red", width=3)
        draw.line(((x_max, y_min), (x_max, y_max)), fill="red", width=3)
        draw.line(((x_min, y_max), (x_max, y_max)), fill="red", width=3)
    plt.figure()
    plt.imshow(new_img)


def get_annotation(offset, size_mask):
    annotation = torch.zeros(5)
    annotation[3] = offset[0]
    annotation[4] = offset[0] + size_mask[0]
    annotation[1] = offset[1]
    annotation[2] = offset[1] + size_mask[1]
    print(annotation.dtype)
    return annotation

print("load images")
path_voc = "F:\Pytorch_Deep_RL\VOCtest2007\VOCdevkit\VOC2007"
image_names = np.array(load_images_names_in_data_set('car_test', path_voc))
labels = load_images_labels_in_data_set('car_test', path_voc)
image_names_aero = []
for i in range(len(image_names)):
    if labels[i] == '1':
        image_names_aero.append(image_names[i])
image_names = image_names_aero
images = get_all_images(image_names, path_voc)
print("car_test image:%d" % len(image_names))

Q_NETWORK_PATH = '../models/' + 'voc2012_2007_model'
model = torch.load(Q_NETWORK_PATH)
model_vgg = getVGG_16bn("../models")
model_vgg = model_vgg.cpu()

class_object = 1
steps = 10
res = []
res_step = []
res_annotations = []
for i in range(len(image_names)):
    image_name = image_names[i]
    image = images[i]

    get_annotation = get_bb_of_gt_from_pascal_xml_annotation(image_name, path_voc)
    original_shape = (image.shape[0], image.shape[1])
    classes_get_objects = get_ids_objects_from_annotation(get_annotation)
    get_masks = generate_bounding_box_from_annotation(get_annotation, image.shape)

    region_image = image
    size_mask = image.shape
    region_mask = torch.ones((image.shape[0], image.shape[1]))
    offset = (0, 0)
    history_vector = torch.zeros((4, 6))
    state = get_state(region_image, history_vector, model_vgg)
    done = False

    for step in range(steps):
        qval = model(Variable(state))
        _, predicted = torch.max(qval.data, 1)
        action = predicted[0] + 1
        if action == 6:
            next_state = None
            done = True
        else:
            offset, region_image, size_mask, region_mask = get_crop_image_and_mask(original_shape, offset,region_image, size_mask,action)

            annotations = []
            annotation = get_annotation(offset, size_mask)
            annotations.append(annotation)
            history_vector = update_history_vector(history_vector, action)
            next_state = get_state(region_image, history_vector, model_vgg)

        state = next_state
        if done:
            res_step.append(step)
            res_annotations.append((get_annotation, annotations, image))
            break

    iou = find_max_bounding_box(get_masks, region_mask, classes_get_objects, class_object)
    pos = 0
    reward = qval.data[0, 5]
    if iou > 0.5:
        pos = 1

    res.append((reward, pos))

begin = 170
end = begin + 10
for i in range(begin, end):
    gt_annotation, annotation, image = res_annotations[i]
    draw_bounding_box_1(get_annotation, image)
    draw_bounding_box_1(annotation, image)


y_test = [x[1] for x in res]
y_score = [x[0] for x in res]
y_test = y_test[::-1]
y_score = y_score[::-1]
average_precision = average_precision_score(y_test, y_score)

print('Average precision-recall score: {0:0.2f}'.format(average_precision))
precision, recall, _ = precision_recall_curve(y_test, y_score)
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1])
plt.xlim([0.0, 1])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))

res_step = np.array(res_step) + 1
plt.hist(res_step)
plt.title('Number of regions analyze per object')
plt.xlabel('Number of regions')
