import random
from os.path import expanduser, join

import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import CfgNode
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectools.utils import register_data


def main(config):

    json_root = expanduser(config["base"]["json_root"])
    imgs_root = expanduser(config["base"]["imgs_root"])
    model_root = expanduser(config["base"]["model_root"])

    testing_thresh = float(config["make_predictions"]["testing_thresh"])
    scale = float(config["make_predictions"]["scale"])
    number_of_imgs = int(config["make_predictions"]["number_of_imgs"])

    if not 0 < testing_thresh < 1:
        raise ValueError(f"The testing threshold, {testing_thresh}, must be between 0 and 1.")

    register_data(json_root, imgs_root)

    # Need this datasets line, in order for metadata to have .thing_classes attribute
    datasets = DatasetCatalog.get("training_data") 
    metadata = MetadataCatalog.get("training_data").set(evaluator_type="coco")
    
    # Read the cfg back in:
    with open(join(model_root, "cfg.txt"), "r") as f:
        cfg = f.read()
    # Turn into CfgNode obj:
    cfg = CfgNode.load_cfg(cfg) 

    # Use the weights from the model trained on our custom dataset:
    cfg.MODEL.WEIGHTS = join(model_root, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = testing_thresh   
    # cfg.DATASETS.TEST = ("val_data", ) # should already be saved from train_model.py

    predictor = DefaultPredictor(cfg)

    # Select random images to visualize the prediction results:
    for i,d in enumerate(random.sample(datasets, number_of_imgs)):

        id = d["image_id"]
        img = cv2.imread(d["file_name"])
        out = predictor(img)
        visualizer = Visualizer(img[:, :, ::-1], 
                                metadata=metadata, 
                                scale=scale, 
                                instance_mode=ColorMode)
        visualizer = visualizer.draw_instance_predictions(out["instances"].to("cpu"))        

        cv2.imshow(f"prediction on image {id}", visualizer.get_image()[:, :, ::-1])
        print(f"Press any key to go to the next image ({i+1}/{number_of_imgs}) ...")

        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            print("Quitting ...")
            break

        cv2.destroyAllWindows()