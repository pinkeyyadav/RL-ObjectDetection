import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
from PIL import ImageDraw
from PIL import Image

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Scale((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def getVGG_16bn(path_vgg):
    state_dict = torch.utils.model_zoo.load_url('https://download.pytorch.org/models/vgg16_bn-6c64b313.pth', path_vgg)
    model = torchvision.models.vgg16_bn()
    model.load_state_dict(state_dict)
    model_2 = list(model.children())[0]
    return model_2


def get_conv_feature_for_image(image, model, dtype=torch.FloatTensor):
    im = transform(image)
    im = im.view(1, *im.shape)
    feature = model(Variable(im).type(dtype))
    return feature.data



##

def load_images_labels_in_data_set(data_set_name, path_voc):
    file_path = path_voc + '/ImageSets/Main/' + data_set_name + '.txt'
    f = open(file_path)
    images_names = f.readlines()
    images_names = [x.split(None, 1)[1] for x in images_names]
    images_names = [x.strip('\n').strip(None) for x in images_names]
    return images_names


def load_image_data(path_voc, class_object):
    print("load images" + path_voc)
    image_names = np.array(load_images_names_in_data_set('car_trainval', path_voc))
    labels = load_images_labels_in_data_set('car_trainval', path_voc)
    image_names_class = []
    for i in range(len(image_names)):
        if labels[i] == class_object:
            image_names_class.append(image_names[i])
    image_names = image_names_class
    images = get_all_images(image_names, path_voc)
    print("total image:%d" % len(image_names))
    return image_names, images


def load_images_names_in_data_set(data_set_name, path_voc):
    file_path = path_voc + '/ImageSets/Main/' + data_set_name + '.txt'
    f = open(file_path)
    image_names = f.readlines()
    image_names = [x.strip('\n') for x in image_names]
    return [x.split(None, 1)[0] for x in image_names]


def get_all_images(image_names, path_voc):
    images = []
    for j in range(np.size(image_names)):
        image_name = image_names[j]
        string = path_voc + '/JPEGImages/' + image_name + '.jpg'
        img = Image.open(string)
        images.append(np.array(img))
    return images
