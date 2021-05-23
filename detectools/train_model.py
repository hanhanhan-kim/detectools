from os import makedirs
from os.path import expanduser
from os.path import join 

from detectron2 import model_zoo
from detectron2.config import get_cfg
# from detectron2.engine import DefaultTrainer # I use my CocoTrainer instead
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectools.utils import register_data
from detectools.coco_trainer import CocoTrainer
from torch._C import Value


def main(config):

    json_root = expanduser(config["base"]["json_root"])
    imgs_root = expanduser(config["base"]["imgs_root"])

    learning_rate = float(config["train_model"]["learning_rate"])
    lr_decay_policy = config["train_model"]["lr_decay_policy"]
    max_iter = int(config["train_model"]["max_iter"])
    eval_period = int(config["train_model"]["eval_period"])
    checkpoint_period = int(config["train_model"]["checkpoint_period"])
    model_root = expanduser(config["base"]["model_root"])

    if not 0 < learning_rate < 1:
        raise ValueError(f"The learning rate, {learning_rate}, must be between 0 and 1.")
    for iter_num in lr_decay_policy:
        if not isinstance(iter_num, int):
            raise TypeError(f"The iteration number, {iter_num}, in {lr_decay_policy} must be an integer.")
    if checkpoint_period < eval_period:
        raise ValueError(f"The checkpoint period, {checkpoint_period}, is less than the evaluation period {eval_period}."
                         "It doesn't make sense to save the model more frequently than its evaluation.")

    register_data(json_root, imgs_root)

    # Need this datasets line, in order for metadata to have .thing_classes attribute
    datasets = DatasetCatalog.get("training_data") 
    metadata = MetadataCatalog.get("training_data").set(evaluator_type="coco") 

    # Load Detectron2's default config:
    cfg = get_cfg()

    # Adjust the default config:
    cfg.merge_from_file(model_zoo.get_config_file("PascalVOC-Detection/faster_rcnn_R_50_C4.yaml"))
    cfg.DATASETS.TRAIN = ("training_data",)

    # Set up evaluation:
    # # https://ortegatron.medium.com/training-on-detectron2-with-a-validation-set-and-plot-loss-on-it-to-avoid-overfitting-6449418fbf4e
    cfg.DATASETS.TEST = ("val_data",)  # our val dataset
    cfg.TEST.EVAL_PERIOD = eval_period # will do an evluation once every this many iters on cfg.DATASETS.TEST
    cfg.SOLVER.CHECKPOINT_PERIOD = checkpoint_period # save the model .pth this many iterations

    # API that maps the model zoo URLs (add .yaml to each one): https://detectron2.readthedocs.io/en/latest/_modules/detectron2/model_zoo/model_zoo.html
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_C4_1x.yaml")
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = learning_rate # Pick a good learning rate
    cfg.SOLVER.STEPS = lr_decay_policy # Each element in this list states the iteration no. at which the LR will decay by a factor of 10. If empty, the LR will not decay.
    cfg.SOLVER.MAX_ITER = (max_iter) # Max number of iterations
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (128)  # faster, and good enough for this toy dataset; default is 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(metadata.thing_classes)  # refers to number of classes, e.g. labels. E.g. WBC, RBC, platelet

    cfg.OUTPUT_DIR = model_root

    # Save settings for later; write the string rep of cfg:
    with open(join(model_root, "cfg.txt"), "w") as f:
        f.write(cfg.dump())
    
    # Train the model with the above settings!:
    makedirs(cfg.OUTPUT_DIR, exist_ok=True) # save
    trainer = CocoTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # TODO: Say in docs that I only support coco