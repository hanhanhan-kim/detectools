from os.path import expanduser, join, basename, splitext
from pathlib import Path
import csv
import atexit

import torch
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
    expected_obj_nums = config["analyze_vids"]["expected_obj_nums"]
    score_cutoff = float(config["analyze_vids"]["score_cutoff"])
    vids_root = expanduser(config["analyze_vids"]["vids_root"])
    framerate = int(config["analyze_vids"]["framerate"])

    if not model_pth.endswith(".pth"):
        raise ValueError(f"{basename(model_pth)} must be a '.pth' file.")
    # if model_pth not in model_dir:
    #     raise IOError(f"The selected model, {basename(model_pth)}, is not in "
    #                   f"{basename(model_dir)}. Please pick a model that resides")
    if not 0 < score_cutoff < 1:
        raise ValueError(f"The testing threshold, {score_cutoff}, must be between 0 and 1.")

    vids = [str(path.absolute()) for path in Path(vids_root).rglob("*.mp4")]
    if len(vids) == 0:
        print(f"No .mp4 videos were found in {vids_root} ...")

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
    # # Pick a confidence cutoff # TODO: based on PR curve
    # # TODO: don't have this as a user parameter
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_cutoff 

    print("Generating predictor ...")
    predictor = DefaultPredictor(cfg)

    bgr_palette = [(165, 194, 102),
                   (98, 141, 252),
                   (203, 160, 141),
                   (195, 138, 231),
                   (84, 216, 166),
                   (47, 217, 255),
                   (148, 196, 229),
                   (179, 179, 179)]

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
                
                # # Visualize:
                # visualizer = Visualizer(frame[:, :, ::-1], 
                #                         metadata=metadata,
                #                         scale=1.0, 
                #                         instance_mode=ColorMode)
                # visualizer = visualizer.draw_instance_predictions(detected["instances"].to("cpu"))      
                # detected_img = visualizer.get_image()[:, :, ::-1]
               
                labelled_frame = frame

                # Save the predicted box coords and scores to a dictionary:
                preds = detected['instances'].to('cpu')
                boxes = preds.pred_boxes.tensor.numpy()
                thing_ids = preds.pred_classes.tolist()
                scores = preds.scores.numpy()

                # Get the idxs of each unique thing_id:
                idxs_of_each_thing = {}
                for thing_id in set(thing_ids):
                    idxs_of_each_thing[thing_id] = []
                for i,thing_id in enumerate(thing_ids):
                    idxs_of_each_thing[thing_id].append(i) 

                # Split up the data according to thing_id:
                for i, (thing_id, idxs) in enumerate(idxs_of_each_thing.items()):

                    thing_class = metadata.thing_classes[thing_id]

                    if thing_class in expected_obj_nums:

                        expected_obj_num = expected_obj_nums[thing_class]               
                        num_boxes = scores.size

                        assert expected_obj_num <= num_boxes, \
                            f"You expected {expected_obj_num} {thing_class} in \
                            every frame of the video, according to `expected_obj_nums`, \
                            but only {num_boxes} were found in frame {f}."

                        thing_scores = scores[idxs]
                        thing_boxes = boxes[idxs]

                        # Here, I grab the top n animals according to their score
                        # because by default, `preds` is sorted by their descending scores:
                        for j in range(0, expected_obj_num):
                                                
                            coords = thing_boxes[j]
                            x1 = int(coords[0])
                            y1 = int(coords[1])
                            x2 = int(coords[2])
                            y2 = int(coords[3])
                            score = float(thing_scores[j])

                            labelled_frame = cv2.rectangle(labelled_frame, (x1, y1), (x2, y2), bgr_palette[thing_id], 2)
                            labelled_frame = cv2.putText(labelled_frame, thing_class, (x2-10,y2-40), cv2.FONT_HERSHEY_SIMPLEX, 1, bgr_palette[thing_id], 2)

                            # TODO: Write some sort of simple filtering thing that skips
                            # big jumps ... maybe just store the last bbox coords and compare.
                            # Define big jumps as a proportion of the frame width or average of 
                            # last few frame deltas
                            csv_writer.writerow({col_names[0]: int(f), # frame
                                                col_names[1]: x1, # x1
                                                col_names[2]: y1, # y1
                                                col_names[3]: x2, # x2
                                                col_names[4]: y2, # y2
                                                col_names[5]: score, # score
                                                col_names[6]: thing_class, # thing
                                                col_names[7]: i}) # dummy id

                # Save frame to vid:
                out.write(labelled_frame)

                pbar.set_description(f"Detecting in frame {f+1}/{frame_count}")

    # Clear GPU memory
    torch.cuda.empty_cache()