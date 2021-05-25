"""
Modified from https://github.com/yukkyo/voc2coco
"""

import os
from os.path import basename, expanduser, join, splitext
from pathlib import Path
import shutil
import json
import random
import xml.etree.ElementTree as ET
from typing import Dict, List
from tqdm import tqdm
import re


def make_ids_from_labels(labels: List[str]) -> Dict[str, int]:

    """Convert labels [strs] to a dictionary of labels:IDs {strings:ints}"""

    # IDs start at 1 instead of 0
    ids = list(range(1, len(labels)+1))

    return dict(zip(labels, ids))


def get_ann_paths(ann_root: str) -> List[str]:

    """From a root directory of annotation xmls, return a list of paths to the xmls"""

    ann_paths = [str(path.absolute()) for path in Path(ann_root).rglob("*.xml")] # recursive
    
    return ann_paths


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
                             output_json: str, 
                             extract_num_from_imgid: bool = True):

    if not output_json.endswith(".json"):
        raise ValueError(f"{basename(output_json)} is not a .json file")

    data = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": []
    }

    bnd_id = 1  # START_BOUNDING_BOX_ID, TODO input as args ?

    # Start converting:
    for a_path in tqdm(ann_paths):

        # Read annotation xml
        ann_tree = ET.parse(a_path)
        ann_root = ann_tree.getroot()

        img_info = get_image_info(ann_root=ann_root,
                                  extract_num_from_imgid=extract_num_from_imgid)
        img_id = img_info['id']
        data['images'].append(img_info)

        for obj in ann_root.findall('object'):
            ann = get_coco_annotation_from_obj(obj=obj, labels_and_ids=labels_and_ids)
            ann.update({'image_id': img_id, 'id': bnd_id})
            data['annotations'].append(ann)
            bnd_id = bnd_id + 1

    for label, label_id in labels_and_ids.items():
        category_info = {'supercategory': 'none', 'id': label_id, 'name': label}
        data['categories'].append(category_info)

    with open(output_json, 'w') as f:
        output_json = json.dumps(data)
        f.write(output_json)


def split_data(ann_paths, output_dir: str, train_frac:float=0.8):

    """Split the list of annotation paths into train, val, and test lists""" 

    if not Path(output_dir).is_dir():
        raise IOError(f"{basename(output_dir)} must be a directory.")
    if train_frac <= 0 and train_frac >= 1:
        raise ValueError(f"train_frac, {train_frac}, is not between 0 and 1.")

    # The split is train:val:test
    val_or_test_frac = (1 - train_frac) / 2
    random.shuffle(ann_paths) # Shuffles in place
    border_bw_train_val = int(train_frac * len(ann_paths))
    border_bw_val_test = int(border_bw_train_val + val_or_test_frac * len(ann_paths))

    train_ann_paths = ann_paths[:border_bw_train_val]
    val_ann_paths = ann_paths[border_bw_train_val:border_bw_val_test]
    test_ann_paths = ann_paths[border_bw_val_test:]

    return train_ann_paths, val_ann_paths, test_ann_paths


def main(config):

    root = expanduser(config["base"]["root"])
    imgs_root = expanduser(config["base"]["imgs_root"])
    do_collate = config["base"]["do_collate"]
    ann_root = expanduser(config["voc_to_coco"]["ann_root"])
    labels = config["voc_to_coco"]["labels"]
    train_frac = config["voc_to_coco"]["train_frac"]

    jsons_dir = join(root, "jsons")
    os.makedirs(jsons_dir, exist_ok=False) 

    labels_and_ids = make_ids_from_labels(labels=labels)
    
    # TODO: MAKE OPTIONAL:
    # sort xmls and jpgs, then zip add unique ids
    # COPY to a new dir called collated, a subidr of root

    # This option will flatten the tree for imgs and xmls found in nested directories:
    if do_collate:
        
        collated_dir = join(root, "collated")
        collated_anns_dir = join(collated_dir, "annotations")
        os.makedirs(collated_anns_dir)
        collated_frames_dir = join(collated_dir, "frames")
        os.makedirs(collated_frames_dir)

        ann_paths = sorted([str(path.absolute()) 
                            for path in Path(ann_root).rglob("*.xml")])
        
        img_exts = [".png", ".jpg", ".tiff"] # TODO: add to README.md
        frame_paths = sorted([str(path.absolute()) 
                              for ext in img_exts 
                              for path in Path(imgs_root).rglob(str("*" + ext)) 
                              if ext in img_exts])

        assert(len(ann_paths) == len(frame_paths), 
               "The no. of xml annotations does not equal the no. of original images")

        for i,(xml, frame) in enumerate(zip(ann_paths, frame_paths)):
            
            uniq_xml = f"frame_{i:06d}.xml"
            ext = splitext(frame)[1]
            uniq_frame = f"frame_{i:06d}{ext}"

            xml_dest = join(collated_anns_dir, uniq_xml)
            img_dest = join(collated_frames_dir, uniq_frame) 

            shutil.copyfile(xml, xml_dest)
            shutil.copyfile(frame, img_dest)
        
        ann_root = collated_anns_dir # we don't use collated_frames_dir in this script

    all_ann_paths = get_ann_paths(ann_root=ann_root)
    train_ann_paths, val_ann_paths, test_ann_paths = split_data(ann_paths=all_ann_paths, 
                                                                output_dir=jsons_dir, 
                                                                train_frac=train_frac)
    all_paths = [all_ann_paths, train_ann_paths, val_ann_paths, test_ann_paths]

    output_jsons = ["all.json", "train.json", "val.json", "test.json"]
    output_jsons = [join(jsons_dir, json) for json in output_jsons]


    for paths, output_json, in zip(all_paths, output_jsons):

        convert_xmls_to_cocojson(ann_paths=paths,
                                 labels_and_ids=labels_and_ids,
                                 output_json=output_json,
                                 extract_num_from_imgid=True)

    print(f"All .json files have been written to {jsons_dir}")

    if do_collate:
        print(f"All collated .xml and image files have been copied to {collated_dir}")

if __name__ == '__main__':
    main()