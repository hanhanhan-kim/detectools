import random
from os.path import expanduser, join
from pathlib import Path

import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectools.utils import register_data


def main(config):
    
    root = expanduser(config["base"]["root"])
    imgs_root = expanduser(config["base"]["imgs_root"])
    scale = float(config["see_data"]["scale"])
    number_of_imgs = int(config["see_data"]["number_of_imgs"])
    jsons_dir = join(root, "jsons")

    register_data(jsons_dir, imgs_root)

    # Training data:
    datasets = DatasetCatalog.get("training_data")
    metadata = MetadataCatalog.get("training_data").set(evaluator_type="coco") # TODO: say in docs that I only support coco

    # Show images:
    if number_of_imgs == 0:
        number_of_imgs = len(datasets)
    for i,d in enumerate(random.sample(datasets, number_of_imgs)):

        id = d["image_id"]
        img = cv2.imread(d["file_name"])

        visualizer = Visualizer(img[:, :, ::-1], 
                                metadata=metadata, 
                                scale=scale)
        out = visualizer.draw_dataset_dict(d)

        cv2.imshow(f"image {id}", out.get_image()[:, :, ::-1])
        print(d["file_name"])
        print(f"Press any key to go to the next image ({i+1}/{number_of_imgs}) ...")

        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            print("Quitting ...")
            break

        cv2.destroyAllWindows()