from os.path import expanduser, join, basename, dirname, splitext
from pathlib import Path
import json

import numpy as np
import cv2
from tqdm import tqdm, trange
from detectron2.engine import DefaultPredictor
from detectron2.config import CfgNode
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectools.utils import register_data


def main(config):

    json_root = expanduser(config["base"]["json_root"])
    imgs_root = expanduser(config["base"]["imgs_root"])
    model_root = expanduser(config["base"]["model_root"])

    model_pth = expanduser(config["analyze_vids"]["model_pth"])
    testing_thresh = float(config["analyze_vids"]["testing_thresh"])
    vids_root = expanduser(config["analyze_vids"]["vids_root"])
    framerate = int(config["analyze_vids"]["framerate"])

    if not model_pth.endswith(".pth"):
        raise ValueError(f"{basename(model_pth)} must be a '.pth' file.")
    if model_pth not in model_root:
        raise IOError(f"The selected model, {basename(model_pth)}, is not in "
                      f"{basename(model_root)}. Please pick a model that resides")
    if not 0 < testing_thresh < 1:
        raise ValueError(f"The testing threshold, {testing_thresh}, must be between 0 and 1.")

    vids = [str(path.absolute()) for path in Path(vids_root).rglob("*.h264")]

    register_data(json_root, imgs_root)

    # Need this datasets line, in order for metadata to have .thing_classes attribute
    datasets = DatasetCatalog.get("training_data") 
    metadata = MetadataCatalog.get("training_data") # don't need to eval this time

    # Read the cfg back in:
    with open(join(model_root, "cfg.txt"), "r") as f:
        cfg = f.read()
    # Turn into CfgNode obj:
    cfg = CfgNode.load_cfg(cfg) 

    # Use the weights from our chosen model:
    cfg.MODEL.WEIGHTS = model_pth
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = testing_thresh

    print("Generating predictor ...")
    predictor = DefaultPredictor(cfg)

    for vid in vids:
        
        cap = cv2.VideoCapture(vid)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = trange(frame_count)
        output_vid = f"{splitext(vid)[0]}_detected.mp4"
        output_json = f"{splitext(vid)[0]}_detected.json"

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*"mp4v") 
        out = cv2.VideoWriter(filename=output_vid, 
                                apiPreference=0, 
                                fourcc=fourcc, 
                                fps=int(framerate), 
                                frameSize=(int(cap.get(3)), int(cap.get(4))), 
                                params=None)

        # Use Detectron2 model on each frame in vid:
        all_detection_info = []
        for f,_ in enumerate(pbar):

            ret, frame = cap.read()

            if ret:

                detected = predictor(frame)
                
                # Visualize:
                visualizer = Visualizer(frame[:, :, ::-1], 
                                        metadata=metadata,
                                        scale=1.0, 
                                        instance_mode=ColorMode)
                visualizer = visualizer.draw_instance_predictions(detected["instances"].to("cpu"))      
                detected_img = visualizer.get_image()[:, :, ::-1]

                # Save frame to vid:
                out.write(detected_img)

                # Save the predicted box coords and scores to a dictionary:
                detection_info = {}
                preds = detected['instances'].to('cpu')
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
                detection_info[f"frame_{f:06d}"] = all_boxes

                all_detection_info.append(detection_info)

                pbar.set_description(f"Detecting {len(thing_ids)} objects "
                                     f"in frame {f+1}/{frame_count}")

        # Write the dictionary to a json:
        with open(output_json, "w") as f:
            json.dump(all_detection_info, f)
        print(f"Saved test predictions to {output_json}")