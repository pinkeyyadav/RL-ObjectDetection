from typing import Type

import torch
import torchvision
import torch.optim as optim
from torch.autograd import Variable
from __init__ import *
from annotate import *
from features import *
from reinforcement import *
from collections import namedtuple
import time
import os
import numpy as np
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path_voc = "F:\Pytorch_Deep_RL\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007"

print("load models")

model_vgg = getVGG_16bn("../models")
model_vgg = model_vgg.cpu()
model = get_q_network()
model = model.cpu()

optimizer = optim.Adam(model.parameters(), lr=1e-6)
criterion = nn.MSELoss().cpu()

path_voc = "F:\Pytorch_Deep_RL\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007"
class_object = '1'
image_names, images = load_image_data(path_voc, class_object)

print("car_trainval image:%d" % len(image_names))

FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
ByteTensor = torch.ByteTensor
Tensor = FloatTensor

epsilon = 1.0
BATCH_SIZE = 200
GAMMA = 0.90
CLASS_OBJECT = 1
steps = 5
epochs = 5
memory = ReplayMemory(1000)


def select_action(states):
    if random.random() < epsilon:
        act = np.random.randint(1, 7)
    else:
        qvalue = model(Variable(states))
        _, predicted = torch.max(qvalue.data, 1)
        act = predicted[0] + 1
    return act


Transition: Type[Transition] = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


def optimizer_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
    next_states = [s for s in batch.next_state if s is not None]
    non_final_next_states = Variable(torch.cat(next_states),volatile=True).type(Tensor)
    state_batch = Variable(torch.cat(batch.state)).type(Tensor)
    action_batch = Variable(torch.LongTensor(batch.action).view(-1, 1)).type(LongTensor)
    reward_batch = Variable(torch.FloatTensor(batch.reward).view(-1, 1)).type(Tensor)

    state_action_values = model(state_batch).gather(1, action_batch)

    next_state_values = Variable(torch.zeros(BATCH_SIZE, 1).type(Tensor))
    next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0]

    next_state_values.volatile = False

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = criterion(state_action_values, expected_state_action_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


print('train the Q-network')
for epoch in range(epochs):
    print('epoch: %d' % epoch)
    now = time.time()
    for i in range(len(image_names)):
        image_name = image_names[i]
        image = images[i]
        if i < len(image_names):
            annotation = get_bb_of_gt_from_pascal_xml_annotation(image_name, path_voc)
        classes_gt_objects = get_ids_objects_from_annotation(annotation)
        gt_masks = generate_bounding_box_from_annotation(annotation, image.shape)

        original_shape = (image.shape[0], image.shape[1])
        region_mask = np.ones((image.shape[0], image.shape[1]))
        iou = find_max_bounding_box(gt_masks, region_mask, classes_gt_objects, CLASS_OBJECT)

        region_image = image
        size_mask = original_shape
        offset = (0, 0)
        history_vector = torch.zeros((4, 6))
        state = get_state(region_image, history_vector, model_vgg)
        done = False
        for step in range(steps):

            if iou > 0.5:
                action = 6
            else:
                action = select_action(state)

            if action == 6:
                next_state = None
                reward = get_reward_trigger(iou)
                done = True
            else:
                offset, region_image, size_mask, region_mask = get_crop_image_and_mask(original_shape, offset,region_image, size_mask, action)
                history_vector = update_history_vector(history_vector, action)
                next_state = get_state(region_image, history_vector, model_vgg)

                new_iou = find_max_bounding_box(gt_masks, region_mask, classes_gt_objects, CLASS_OBJECT)
                reward = get_reward_movement(iou, new_iou)
                iou = new_iou
            print('epoch: %d, image: %d, step: %d, reward: %d' % (epoch, i, step, reward))
            #memory.push(state, action - 1, next_state, reward)

            state = next_state

            optimizer_model()
            if done:
                break
    if epsilon > 0.1:
        epsilon -= 0.1
    time_cost = time.time() - now
    print('epoch = %d, time_cost = %.4f' % (epoch, time_cost))

Q_NETWORK_PATH = '../models/' + 'voc2012_2007_model'
torch.save(model, Q_NETWORK_PATH)
print('Complete')
