import os
from os.path import expanduser, join

from detectron2.engine import DefaultPredictor
from detectron2.config import CfgNode

from detectools.utils import register_data


def main(config):

    json_root = expanduser(config["base"]["json_root"])
    imgs_root = expanduser(config["base"]["imgs_root"])
    model_root = expanduser(config["base"]["model_root"])

    testing_thresh = float(config["make_predictions"]["testing_thresh"])

    if not 0 < testing_thresh < 1:
        raise ValueError(f"The testing threshold, {testing_thresh}, must be between 0 and 1.")

    register_data(json_root, imgs_root)
    
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

    # TODO: Select random images to visualize the prediction results