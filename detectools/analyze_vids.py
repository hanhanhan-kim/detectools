from os.path import expanduser, join, basename, dirname, splitext
from os import makedirs
from pathlib import Path
import csv
import atexit

import numpy as np
import cv2
from tqdm import trange
from detectron2.engine import DefaultPredictor
from detectron2.config import CfgNode
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectools.utils import register_data


def main(config):

    root = expanduser(config["base"]["root"])
    imgs_root = expanduser(config["base"]["imgs_root"])
    jsons_dir = join(root, "jsons")
    model_dir = join(root, "outputs")

    model_pth = expanduser(config["analyze_vids"]["model_pth"])
    score_cutoff = float(config["analyze_vids"]["score_cutoff"])
    vids_root = expanduser(config["analyze_vids"]["vids_root"])
    framerate = int(config["analyze_vids"]["framerate"])

    if not model_pth.endswith(".pth"):
        raise ValueError(f"{basename(model_pth)} must be a '.pth' file.")
    if model_pth not in model_dir:
        raise IOError(f"The selected model, {basename(model_pth)}, is not in "
                      f"{basename(model_dir)}. Please pick a model that resides")
    if not 0 < score_cutoff < 1:
        raise ValueError(f"The testing threshold, {score_cutoff}, must be between 0 and 1.")

    vids = [str(path.absolute()) for path in Path(vids_root).rglob("*.h264")]

    register_data(jsons_dir, imgs_root)

    # Need the `datasets =` line, in order for metadata to have the 
    # .thing_classes attrib. I don't really use these two lines, I 
    # only call them so I can get the .thing_classes attrib off 
    # `metadata`. So, it doesn't matter if I use "training_data" as 
    # my arg or some other registered dataset, for these two calls:
    datasets = DatasetCatalog.get("training_data") 
    metadata = MetadataCatalog.get("training_data")

    # Read the cfg back in:
    with open(join(model_dir, "cfg.txt"), "r") as f:
        cfg = f.read()
    # Turn into CfgNode obj:
    cfg = CfgNode.load_cfg(cfg) 

    # Use the weights from our chosen model:
    cfg.MODEL.WEIGHTS = model_pth
    # Pick a confidence cutoff based on PR curve
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_cutoff

    print("Generating predictor ...")
    predictor = DefaultPredictor(cfg)

    for vid in vids:
        
        cap = cv2.VideoCapture(vid)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = trange(frame_count)
        output_vid = f"{splitext(vid)[0]}_detected.mp4"

        output_csv = f"{splitext(vid)[0]}_detected.csv"
        csv_file_handle = open(output_csv, "w", newline="")
        atexit.register(csv_file_handle.close) 
        col_names = ["frame", "x1", "y1", "x2", "y2", "score", "thing", "dummy_id"]
        csv_writer = csv.DictWriter(csv_file_handle, fieldnames=col_names)
        csv_writer.writeheader()

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

                    csv_writer.writerow({col_names[0]: int(f), # frame
                                         col_names[1]: int(coords[0][0]), # x1
                                         col_names[2]: int(coords[0][1]), # y1
                                         col_names[3]: int(coords[0][2]), # x2
                                         col_names[4]: int(coords[0][3]), # y2
                                         col_names[5]: score, # score
                                         col_names[6]: thing_class, # thing
                                         col_names[7]: i}) # dummy id

                pbar.set_description(f"Detecting {len(thing_ids)} objects "
                                     f"in frame {f+1}/{frame_count}")