import random
from os.path import expanduser, join, basename
from os import makedirs
import csv
import atexit

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

    scale = float(config["eval_model"]["scale"])
    number_of_imgs = int(config["eval_model"]["number_of_imgs"])
    do_show = config["eval_model"]["do_show"]

    register_data(json_root, imgs_root)

    # Need this datasets line, in order for metadata to have .thing_classes attribute
    datasets = DatasetCatalog.get("val_data") 
    metadata = MetadataCatalog.get("val_data").set(evaluator_type="coco")
    
    # Read the cfg back in:
    with open(join(model_root, "cfg.txt"), "r") as f:
        cfg = f.read()
    # Turn into CfgNode obj:
    cfg = CfgNode.load_cfg(cfg) 

    # Use the weights from the model trained on our custom dataset:
    cfg.MODEL.WEIGHTS = join(model_root, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01 # make small so I can make PR curve for broad range of scores
    # cfg.DATASETS.TEST = ("val_data", ) # should already be saved from train_model.py

    print("Generating predictor ...")
    predictor = DefaultPredictor(cfg)

    # For saving images with predicted labels:
    makedirs(join(model_root, "test_pred_imgs"), exist_ok=True)

    # For saving detection predictions as csv:
    output_csv = join(model_root, "all_test_preds.csv")
    csv_file_handle = open(output_csv, "w", newline="")
    atexit.register(csv_file_handle.close) 
    col_names = ["img", "x1", "y1", "x2", "y2", "score", "thing","dummy_id"]
    csv_writer = csv.DictWriter(csv_file_handle, fieldnames=col_names)
    csv_writer.writeheader()

    # Select random images to visualize and save the prediction results:
    # TODO!!!!!!!!!!!: Save a random set of frames, but run prediction on ALL of them.
    for i,d in enumerate(random.sample(datasets, number_of_imgs)):

        print(f"Predicting on image {i+1} of {number_of_imgs} ...")

        id = d["image_id"]
        img = cv2.imread(d["file_name"])
        detected = predictor(img)
        
        # Visualize:
        visualizer = Visualizer(img[:, :, ::-1], 
                                metadata=metadata, 
                                scale=scale, 
                                instance_mode=ColorMode)
        visualizer = visualizer.draw_instance_predictions(detected["instances"].to("cpu"))        
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

        # Stream the predicted box coords and scores to a csv:
        preds = detected['instances'].to('cpu')
        boxes = preds.pred_boxes
        thing_ids = preds.pred_classes.tolist()
        scores = preds.scores
        num_boxes = np.array(scores.size())[0]

        for i in range(0, num_boxes):
            coords = boxes[i].tensor.numpy()    	
            score = float(scores[i].numpy())
            thing_id = thing_ids[i] # is int
            thing_class = metadata.thing_classes[thing_id]

            csv_writer.writerow({col_names[0]: basename(d["file_name"]),
                                 col_names[1]: int(coords[0][0]), # x1
                                 col_names[2]: int(coords[0][1]), # y1
                                 col_names[3]: int(coords[0][2]), # x2
                                 col_names[4]: int(coords[0][3]), # y2
                                 col_names[5]: score, # score
                                 col_names[6]: thing_class, # thing
                                 col_names[7]: i}) # dummy id