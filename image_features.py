import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Scale((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


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
