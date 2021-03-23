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
from annotate import *
from features import *
from reinforcement import *
import logging
import time
import os

path_voc = "F:\Pytorch_Deep_RL\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007"

print("load models")

model_vgg = getVGG_16bn("../models")
model_vgg = model_vgg.cpu()
model = get_q_network()
model = model.cpu()

optimizer = optim.Adam(model.parameters(), lr=1e-6)
criterion = nn.MSELoss().cpu()

print("load images")

path_voc = "F:\Pytorch_Deep_RL\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007"
image_names = np.array(load_images_names_in_data_set('car_trainval', path_voc))
labels = load_images_labels_in_data_set('car_trainval', path_voc)
image_names_aero = []
for i in range(len(image_names)):
    if labels[i] == '1':
        image_names_aero.append(image_names[i])
image_names = image_names_aero
images = get_all_images(image_names, path_voc)

print("car_trainval image:%d" % len(image_names))

from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
ByteTensor = torch.ByteTensor
Tensor = FloatTensor


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


epsilon = 1.0
BATCH_SIZE = 200
GAMMA = 0.90
CLASS_OBJECT = 1
steps = 5
epochs = 5
memory = ReplayMemory(1000)


def select_action(state):
    if np.random.uniform(1,7) < epsilon:
        action = np.random.randint(1, 7)
    else:
        qval = model(Variable(state))
        _, predicted = torch.max(qval.data, 1)
        action = predicted[0] + 1
        return action


def optimizer_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
    next_states = [s for s in batch.next_state if s is not None]
    non_final_next_states = Variable(torch.cat(next_states),
                                     volatile=True).type(Tensor)
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

Q_NETWORK_PATH = '../models/' + 'one_object_model_2'
torch.save(model, Q_NETWORK_PATH)
print('Complete')