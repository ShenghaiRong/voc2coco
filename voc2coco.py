import os
import argparse
import json
import xml.etree.ElementTree as ET
from typing import Dict
from tqdm import tqdm
from pycocotools import mask
import numpy as np
from skimage import measure
from itertools import groupby
from PIL import Image
import imageio


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def get_label2id(labels_path: str) -> Dict[str, int]:
    """id is 1 start"""
    with open(labels_path, 'r') as f:
        labels_str = f.read().split()
    labels_ids = list(range(1, len(labels_str)+1))
    return dict(zip(labels_str, labels_ids))


def compress_range(arr):
    uniques = np.unique(arr)
    maximum = np.max(uniques)

    d = np.zeros(maximum+1, np.int32)
    d[uniques] = np.arange(uniques.shape[0])

    out = d[arr]
    return out - np.min(out)


def sparse_to_one_hot(arr, maximum_val=None, dtype=np.bool):

    sparse_integers = compress_range(arr)

    if maximum_val is None:
        maximum_val = np.max(sparse_integers) + 1

    src_shape = sparse_integers.shape

    flat_src = np.reshape(sparse_integers, [-1])
    src_size = flat_src.shape[0]

    one_hot = np.zeros((maximum_val, src_size), dtype)
    one_hot[flat_src, np.arange(src_size)] = 1

    one_hot = np.reshape(one_hot, [maximum_val] + list(src_shape))

    return one_hot

def get_mask(mask_path, img_name):
    mask = imageio.imread(mask_path + img_name + '.png')
    binary_mask = sparse_to_one_hot(mask)
    return binary_mask


def resize_binary_mask(array, new_size):
    image = Image.fromarray(array.astype(np.uint8)*255)
    image = image.resize(new_size)
    return np.asarray(image).astype(np.bool_)

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour

def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
                counts.append(0)
        counts.append(len(list(elements)))

    return rle

def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation
    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation 
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons


def get_coco_annotation_from_xml(obj, label2id):
    label = obj.findtext('name')
    assert label in label2id, f"Error: {label} is not in label2id !"
    category_id = label2id[label]
    bndbox = obj.find('bndbox')
    xmin = int(float(bndbox.findtext('xmin'))) - 1
    ymin = int(float(bndbox.findtext('ymin'))) - 1
    xmax = int(float(bndbox.findtext('xmax')))
    ymax = int(float(bndbox.findtext('ymax')))
    assert xmax > xmin and ymax > ymin, f"Box size error !: (xmin, ymin, xmax, ymax): {xmin, ymin, xmax, ymax}"
    o_width = xmax - xmin
    o_height = ymax - ymin
    ann = {
        'area': o_width * o_height,
        'iscrowd': 0,
        'bbox': [xmin, ymin, o_width, o_height],
        'category_id': category_id,
        'ignore': 0,
        'segmentation': []  # This script is not for segmentation
    }
    return ann

def get_coco_annotation_from_mask(seg, binary_mask, image_size, 
                                 is_crowd=0, bounding_box=None):
    label = np.unique(seg * binary_mask)
    if 255 in label:
        label = label[:-1]
    if label[-1] == 0:
        return None
    assert len(label) == 2
    category_id = label[1]

    if image_size is not None:
        binary_mask = resize_binary_mask(binary_mask, image_size)

    binary_mask_encoded = mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))

    area = mask.area(binary_mask_encoded)
    if area < 1:
        return None

    if bounding_box is None:
        bounding_box = mask.toBbox(binary_mask_encoded)

    if is_crowd == 1:
        is_crowd = 1
        segmentation = binary_mask_to_rle(binary_mask)
    else :
        is_crowd = 0
        segmentation = binary_mask_to_polygon(binary_mask, tolerance=0)
        if not segmentation:
            return None
    ann = {
        'area': bounding_box[2] * bounding_box[3],
        'iscrowd': 0,
        #'bbox': [xmin, ymin, width, height],
        'bbox': bounding_box,
        'category_id': category_id,
        'ignore': 0,
        'segmentation': segmentation
    }
    return ann

def get_coco_annotation_from_npy(binary_mask, image_size, category_id,
                                 is_crowd=0, bounding_box=None):

    if image_size is not None:
        binary_mask = resize_binary_mask(binary_mask, image_size)

    binary_mask_encoded = mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))

    area = mask.area(binary_mask_encoded)
    if area < 1:
        return None

    if bounding_box is None:
        bounding_box = mask.toBbox(binary_mask_encoded)

    if is_crowd == 1:
        is_crowd = 1
        segmentation = binary_mask_to_rle(binary_mask)
    else :
        is_crowd = 0
        segmentation = binary_mask_to_polygon(binary_mask, tolerance=0)
        if not segmentation:
            return None
    ann = {
        'area': bounding_box[2] * bounding_box[3],
        'iscrowd': 0,
        #'bbox': [xmin, ymin, o_width, o_height],
        'bbox': bounding_box,
        'category_id': category_id,
        'ignore': 0,
        'segmentation': segmentation
    }
    return ann


def convert_voc_ann_to_cocojson(image_list_path: str,
                                image_path: str,
                                annotation_path: str,
                                label2id: Dict[str, int],
                                output_jsonpath: str,
                                voc_ins_label_style: str='png',
                                seg_path: str=None):
    output_json_dict = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": []
    }
    image_list = np.loadtxt(image_list_path, dtype=np.str_)
    bnd_id = 1  # START_BOUNDING_BOX_ID 
    img_id = 1  # START_IMAGE_ID
    print('Start converting !')
    for img_name in tqdm(image_list):
        img_path = os.path.join(image_path, img_name+'.jpg')
        img = imageio.imread(img_path)
        height, width, _ = img.shape

        img_info = {
        'file_name': img_name + '.jpg',
        'height': height,
        'width': width,
        'id': img_id
        }
        img_size = [width, height]
        output_json_dict['images'].append(img_info)
        if voc_ins_label_style == 'png':
            mask = get_mask(annotation_path, img_name)
            seg = imageio.imread(seg_path + img_name + '.png')
            for mask_id in range(1, mask.shape[0]):
                ann_info = get_coco_annotation_from_mask(seg=seg, 
                                                         binary_mask=mask[mask_id], 
                                                         image_size=img_size)
                if ann_info is None:
                    continue
                ann_info.update({'image_id': img_id, 'id': bnd_id})
                output_json_dict['annotations'].append(ann_info)
                bnd_id = bnd_id + 1
        elif voc_ins_label_style == 'npy':
            ann = np.load(os.path.join(annotation_path, img_name+'.npy'), allow_pickle=True).item()
            for score, mask, class_id in zip(ann['score'], ann['mask'], ann['class']):
                if score < 1e-5:
                    continue
                ann_info = get_coco_annotation_from_npy(mask, img_size, class_id+1)
                if ann_info is None:
                    continue
                ann_info.update({'image_id': img_id, 'id': bnd_id})
                output_json_dict['annotations'].append(ann_info)
                bnd_id += 1
        else:
            assert voc_ins_label_style == 'xml'
            a_path = os.path.join(annotation_path, img_name + '.xml')
            ann_tree = ET.parse(a_path)
            ann_root = ann_tree.getroot()
            for obj in ann_root.findall('object'):
                ann = get_coco_annotation_from_xml(obj=obj, label2id=label2id)
                ann.update({'image_id': img_id, 'id': bnd_id})
                output_json_dict['annotations'].append(ann)
                bnd_id = bnd_id + 1

        img_id += 1


    for label, label_id in label2id.items():
        category_info = {'supercategory': 'none', 'id': label_id, 'name': label}
        output_json_dict['categories'].append(category_info)

    with open(output_jsonpath, 'w') as f:
        output_json = json.dumps(output_json_dict, cls=NumpyEncoder)
        f.write(output_json)



def main():
    parser = argparse.ArgumentParser(
        description='This script support converting voc instance annotations to coco format json')
    parser.add_argument('--image_list_dir', type=str, 
                        default='/data/rongsh/data/VOCSBD/VOC2012/ImageSets/Segmentation/train_aug.txt',
                        help='path to image list files directory.')
    parser.add_argument('--image_dir', type=str, 
                        default='/data/rongsh/data/VOCSBD/VOC2012/JPEGImages/',
                        help='path to image files directory.')
    parser.add_argument('--ann_dir', type=str,
                        default='/data/rongsh/data/VOCSBD/VOC2012/VOCSBD_GT_inst/',
                        help='path to annotation files directory.')
    parser.add_argument('--labels', type=str, 
                        default='/data/rongsh/data/VOCSBD/VOC2012/labels.txt',
                        help='path to label list.')
    parser.add_argument('--output', type=str, default='voc_cocostyle.json', help='path to output json file')
    parser.add_argument('--voc_ins_label_style', type=str, default='png', 
                        help='extension of annotation file')
    parser.add_argument('--seg_dir', type=str,
                        default='/data/rongsh/data/VOCSBD/VOC2012/SegmentationClassAug/',
                        help='path to ground truth segmentation files directory.')

   # voc_ins_label_style = 'xml'
   # base_dir = '/data/rongsh/data/VOCSBD/VOC2012/'
   # img_list_dir = base_dir + 'ImageSets/Segmentation/train_aug.txt'
   # img_dir = base_dir + 'JPEGImages/'
   # if voc_ins_label_style == 'png':
   #     ann_dir = base_dir + 'VOCSBD_GT_inst/'
   #     seg_dir = base_dir + 'SegmentationClassAug/'
   # elif voc_ins_label_style == 'npy':
   #     ann_dir = '/home/rongsh/pytorch/wsis/output/irn_best/ins_seg_out_dir/'
   #     seg_dir = None
   # else:
   #     assert voc_ins_label_style == 'xml'
   #     # xml annotations do not include instance mask
   #     ann_dir = base_dir + 'Annotations/'
   #     seg_dir = None
   # output = base_dir + 'voc2012_train_aug_cocostyle_xml_test.json'
   # labels = base_dir + 'labels.txt'

    args = parser.parse_args()
    labels = args.labels
    label2id = get_label2id(labels_path=labels)
    convert_voc_ann_to_cocojson(image_list_path=args.image_list_dir,
                                image_path=args.image_dir,
                                annotation_path=args.ann_dir,
                                label2id=label2id,
                                output_jsonpath=args.output,
                                voc_ins_label_style=args.voc_ins_label_style,
                                seg_path=args.seg_dir)


if __name__ == '__main__':
    main()