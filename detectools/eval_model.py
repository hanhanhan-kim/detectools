import random
from os.path import expanduser, join, basename
from os import makedirs
import json

import numpy as np
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

    testing_thresh = float(config["eval_model"]["testing_thresh"])
    scale = float(config["eval_model"]["scale"])
    number_of_imgs = int(config["eval_model"]["number_of_imgs"])
    do_show = config["eval_model"]["do_show"]

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

    print("Generating predictor ...")
    predictor = DefaultPredictor(cfg)

    # For saving images with predicted labels:
    makedirs(join(model_root, "test_pred_imgs"), exist_ok=True)

    # Select random images to visualize and save the prediction results:
    for i,d in enumerate(random.sample(datasets, number_of_imgs)):

        id = d["image_id"]
        img = cv2.imread(d["file_name"])
        outputs = predictor(img)
        
        # Visualize:
        visualizer = Visualizer(img[:, :, ::-1], 
                                metadata=metadata, 
                                scale=scale, 
                                instance_mode=ColorMode)
        visualizer = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))        
        pred_img = visualizer.get_image()[:, :, ::-1]

        if do_show:

            cv2.imshow(f"prediction on image {id}", pred_img)
            print(f"Press any key to go to the next image ({i+1}/{number_of_imgs}) ...")

            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                print("Quitting ...")
                break

        cv2.destroyAllWindows()

        cv2.imwrite(join(model_root, "test_pred_imgs", ("predicted_" + basename(d["file_name"]))), pred_img)

        # Save the predicted box coords and scores to a dictionary:
        test_preds = {}
        preds = outputs['instances'].to('cpu')
        boxes = preds.pred_boxes
        thing_ids = preds.pred_classes.tolist()
        scores = preds.scores
        num_boxes = np.array(scores.size())[0]
        all_boxes = []

        for i in range(0, num_boxes):
            coords = boxes[i].tensor.numpy()    	
            score = float(scores[i].numpy())
            thing_id = thing_ids[i] # is int
            thing_class = metadata.thing_classes[thing_id]
            all_boxes.append([int(coords[0][0]), # x1
                              int(coords[0][1]), # y1
                              int(coords[0][2]), # x2
                              int(coords[0][3]), # y2
                              score,
                              thing_class])

        test_preds[d["file_name"]] = all_boxes

    # Write the dictionary to a json:
    output_json = join(model_root, "test_preds.json")
    with open(output_json, "w") as f:
        json.dump(test_preds, f)
    print(f"Saved test predictions to {output_json}")