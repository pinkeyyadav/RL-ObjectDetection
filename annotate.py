import xml.etree.ElementTree as ET
import numpy as np
import cv2


def get_bb_of_gt_from_pascal_xml_annotation(xml_name, voc_path):
    string = voc_path + '/Annotations/' + xml_name + '.xml'
    tree = ET.parse(string)
    root = tree.getroot()
    names = []
    x_min = []
    x_max = []
    y_min = []
    y_max = []
    for child in root:
        if child.tag == 'object':
            for child2 in child:
                if child2.tag == 'name':
                    names.append(child2.text)
                elif child2.tag == 'bndbox':
                    for child3 in child2:
                        if child3.tag == 'xmin':
                            x_min.append(child3.text)
                        elif child3.tag == 'xmax':
                            x_max.append(child3.text)
                        elif child3.tag == 'ymin':
                            y_min.append(child3.text)
                        elif child3.tag == 'ymax':
                            y_max.append(child3.text)
    category_and_bb = np.zeros([np.size(names), 5])
    for i in range(np.size(names)):
        category_and_bb[i][0] = get_id_of_class_name(names[i])
        category_and_bb[i][1] = x_min[i]
        category_and_bb[i][2] = x_max[i]
        category_and_bb[i][3] = y_min[i]
        category_and_bb[i][4] = y_max[i]
    return category_and_bb


def get_all_annotations(image_names, voc_path):
    annotations = []
    for i in range(np.size(image_names)):
        image_name = image_names[i]
        annotations.append(get_bb_of_gt_from_pascal_xml_annotation(image_name, voc_path))
    return annotations


def generate_bounding_box_from_annotation(annotation, image_shape):
    length_annotation = annotation.shape[0]
    masks = np.zeros([image_shape[0], image_shape[1], length_annotation])
    for i in range(0, length_annotation):
        masks[int(annotation[i, 3]):int(annotation[i, 4]), int(annotation[i, 1]):int(annotation[i, 2]), i] = 1
    return masks


def get_ids_objects_from_annotation(annotation):
    return annotation[:, 0]


def get_id_of_class_name (class_name):
    if class_name == 'aeroplane':
        return 1
    elif class_name == 'bicycle':
        return 2
    elif class_name == 'bird':
        return 3
    elif class_name == 'boat':
        return 4
    elif class_name == 'bottle':
        return 5
    elif class_name == 'bus':
        return 6
    elif class_name == 'car':
        return 7
    elif class_name == 'cat':
        return 8
    elif class_name == 'chair':
        return 9
    elif class_name == 'cow':
        return 10
    elif class_name == 'diningtable':
        return 11
    elif class_name == 'dog':
        return 12
    elif class_name == 'horse':
        return 13
    elif class_name == 'motorbike':
        return 14
    elif class_name == 'person':
        return 15
    elif class_name == 'pottedplant':
        return 16
    elif class_name == 'sheep':
        return 17
    elif class_name == 'sofa':
        return 18
    elif class_name == 'train':
        return 19
    elif class_name == 'tvmonitor':
        return 20


##

scale_subregion = float(3) / 4
scale_mask = float(1) / (scale_subregion * 4)


def calculate_iou(img_mask, get_mask):
    get_mask *= 1.0
    img_and = cv2.bitwise_and(img_mask, get_mask)
    img_or = cv2.bitwise_or(img_mask, get_mask)
    j = np.count_nonzero(img_and)
    i = np.count_nonzero(img_or)
    iou = float(float(j) / float(i))
    return iou


def calculate_overlapping(img_mask, get_mask):
    get_mask *= 1.0
    img_and = cv2.bitwise_and(img_mask, get_mask)
    j = np.count_nonzero(img_and)
    i = np.count_nonzero(get_mask)
    overlap = float(float(j) / float(i))
    return overlap


def follow_iou(gt_masks, region_mask, classes_get_objects, class_object, last_matrix):
    results = np.zeros([np.size(array_classes_get_objects), 1])
    for k in range(np.size(classes_get_objects)):
        if classes_get_objects[k] == class_object:
            get_mask = get_masks[:, :, k]
            iou = calculate_iou(region_mask, get_mask)
            results[k] = iou
    index = np.argmax(results)
    new_iou = results[index]
    iou = last_matrix[index]
    return iou, new_iou, results, index


def find_max_bounding_box(get_masks, region_mask, classes_get_objects, class_object):
    _, _, n = get_masks.shape
    max_iou = 0.0
    for k in range(n):
        if classes_get_objects[k] != class_object:
            continue
        gt_mask = get_masks[:, :, k]
        iou = calculate_iou(region_mask, gt_mask)
        if max_iou < iou:
            max_iou = iou
    return max_iou


def get_crop_image_and_mask(original_shape, offset, region_image, size_mask, action):

    region_mask = np.zeros(original_shape)
    size_mask = (int(size_mask[0] * scale_subregion), int(size_mask[1] * scale_subregion))
    if action == 1:
        offset_aux = (0, 0)
    elif action == 2:
        offset_aux = (0, int(size_mask[1] * scale_mask))
        offset = (offset[0], offset[1] + int(size_mask[1] * scale_mask))
    elif action == 3:
        offset_aux = (int(size_mask[0] * scale_mask), 0)
        offset = (offset[0] + int(size_mask[0] * scale_mask), offset[1])
    elif action == 4:
        offset_aux = (int(size_mask[0] * scale_mask),
                      int(size_mask[1] * scale_mask))
        offset = (offset[0] + int(size_mask[0] * scale_mask),
                  offset[1] + int(size_mask[1] * scale_mask))
    elif action == 5:
        offset_aux = (int(size_mask[0] * scale_mask / 2),
                      int(size_mask[0] * scale_mask / 2))
        offset = (offset[0] + int(size_mask[0] * scale_mask / 2),
                  offset[1] + int(size_mask[0] * scale_mask / 2))
    region_image = region_image[offset_aux[0]:offset_aux[0] + size_mask[0],offset_aux[1]:offset_aux[1] + size_mask[1]]
    region_mask[offset[0]:offset[0] + size_mask[0], offset[1]:offset[1] + size_mask[1]] = 1
    return offset, region_image, size_mask, region_mask













