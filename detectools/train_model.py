from os import makedirs, getcwd
from os.path import expanduser, dirname
import os

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, DefaultPredictor

from detectools.utils import register_data

def main(config):

    json_dir = expanduser(config["voc_to_coco"]["output_dir"]) # from two commands ago
    imgs_root = expanduser(config["see_data"]["imgs_root"]) # from previous command

    register_data(json_dir, imgs_root)

    # Load Detectron2's default config:
    cfg = get_cfg()

    # TODO: Mention in docs that I"m only supporting 1 model with this wrapper right now:
    # Adjust the default config:
    cfg.merge_from_file(model_zoo.get_config_file("PascalVOC-Detection/faster_rcnn_R_50_C4.yaml"))
    cfg.DATASETS.TRAIN = ("training_data",)

    # TODO: Figure out which of the below to put in config.yaml as params:

    # Set up evaluation:
    # # https://ortegatron.medium.com/training-on-detectron2-with-a-validation-set-and-plot-loss-on-it-to-avoid-overfitting-6449418fbf4e
    cfg.DATASETS.TEST = ("val_data",)  
    cfg.TEST.EVAL_PERIOD = 100 

    # Can't seem to initialize these particular weights from model zoo, even though I'm referencing the API
    # API that maps the model zoo URLs: https://detectron2.readthedocs.io/en/latest/_modules/detectron2/model_zoo/model_zoo.html
    # The weights were downloaded from https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md
    # TODO: Mention in docs that I"m only supporting 1 model with this wrapper right now:
    cfg.MODEL.WEIGHTS = os.path.join(dirname(getcwd()), "detectools/detectools/pretrained_weights/model_final_721ade.pkl")

    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.02 # Pick a good loss rate
    cfg.SOLVER.MAX_ITER = (300)  
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (128)  # faster, and good enough for this toy dataset; default is 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # refers to number of classes, e.g. labels. E.g. WBC, RBC, platelet

    # TODO: cfg.OUTPUT_DIR set this to a good location

    # Train the model with the above settings:
    makedirs(cfg.OUTPUT_DIR, exist_ok=True) # save
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()