"""
Modified from https://github.com/yukkyo/voc2coco
"""

import os
from os.path import basename, expanduser
from pathlib import Path
import json
import xml.etree.ElementTree as ET
from typing import Dict, List
from tqdm import tqdm
import re


def make_ids_from_labels(labels: List[str]) -> Dict[str, int]:

    """Convert labels [strs] to a dictionary of labels:IDs {strings:ints}"""

    # IDs start at 1 instead of 0
    ids = list(range(1, len(labels)+1))

    return dict(zip(labels, ids))


def get_ann_paths(root: str) -> List[str]:

    """From a root directory of annotation xmls, return a list of paths to the xmls"""

    xmls = [str(path.absolute()) for path in Path(root).rglob("*.xml")] # recursive
    
    return xmls


def get_image_info(ann_root, extract_num_from_imgid=True):

    path = ann_root.findtext('path')

    if path is None:
        filename = ann_root.findtext('filename')
    else:
        filename = os.path.basename(path)

    img_name = os.path.basename(filename)
    img_id = os.path.splitext(img_name)[0]

    if extract_num_from_imgid and isinstance(img_id, str):
        img_id = int(re.findall(r'\d+', img_id)[0])

    size = ann_root.find('size')
    width = int(size.findtext('width'))
    height = int(size.findtext('height'))

    image_info = {
        'file_name': filename,
        'height': height,
        'width': width,
        'id': img_id
    }

    return image_info


def get_coco_annotation_from_obj(obj, labels_and_ids):

    label = obj.findtext('name')
    assert label in labels_and_ids, f"Error: {label} is not in labels_and_ids !"
    category_id = labels_and_ids[label]

    bndbox = obj.find('bndbox')
    xmin = int(bndbox.findtext('xmin')) - 1
    ymin = int(bndbox.findtext('ymin')) - 1
    xmax = int(bndbox.findtext('xmax'))
    ymax = int(bndbox.findtext('ymax'))
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


def convert_xmls_to_cocojson(ann_paths: List[str],
                             labels_and_ids: Dict[str, int],
                             output_jsonpath: str,
                             extract_num_from_imgid: bool = True):

    output_json_dict = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": []
    }

    bnd_id = 1  # START_BOUNDING_BOX_ID, TODO input as args ?
    print('Start converting !')

    for a_path in tqdm(ann_paths):
        # Read annotation xml
        ann_tree = ET.parse(a_path)
        ann_root = ann_tree.getroot()

        img_info = get_image_info(ann_root=ann_root,
                                  extract_num_from_imgid=extract_num_from_imgid)
        img_id = img_info['id']
        output_json_dict['images'].append(img_info)

        for obj in ann_root.findall('object'):
            ann = get_coco_annotation_from_obj(obj=obj, labels_and_ids=labels_and_ids)
            ann.update({'image_id': img_id, 'id': bnd_id})
            output_json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1

    for label, label_id in labels_and_ids.items():
        category_info = {'supercategory': 'none', 'id': label_id, 'name': label}
        output_json_dict['categories'].append(category_info)

    with open(output_jsonpath, 'w') as f:
        output_json = json.dumps(output_json_dict)
        f.write(output_json)


def main(config):

    root = expanduser(config["voc_to_coco"]["root"])
    labels = config["voc_to_coco"]["labels"]
    output = expanduser(config["voc_to_coco"]["path_to_output"])

    if not output.endswith(".json"):
        raise ValueError(f"{basename(output)} must end in '.json'")

    labels_and_ids = make_ids_from_labels(labels=labels)
    ann_paths = get_ann_paths(root=root)

    convert_xmls_to_cocojson(ann_paths=ann_paths,
                             labels_and_ids=labels_and_ids,
                             output_jsonpath=output,
                             extract_num_from_imgid=True
    )


if __name__ == '__main__':
    main()