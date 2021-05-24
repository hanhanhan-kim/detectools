from os.path import expanduser, basename
from os.path import join as path_join
from pathlib import Path

from detectron2.data.datasets import register_coco_instances


def register_data(json_dir, imgs_root):

    """
    Register COCO-formatted json data.
    
    Parameters:
    -----------
    json_dir (str): The directory that contains the `train.json` and `val.json` files that output 
        from the `main()` of the the `voc_to_coco.py` module. 
    imgs_root (str): The directory that contains the original images. 

    Returns:
    --------
    Void. Just registers the data.     
    """

    json_dir = expanduser(json_dir) # json_dir = expanduser(config["voc_to_coco"]["output_dir"])
    imgs_root = expanduser(imgs_root)

    if not Path(json_dir).is_dir():
        raise ValueError(f"{basename(json_dir)} is not a directory")
    if not Path(imgs_root).is_dir():
        raise ValueError(f"{basename(imgs_root)} is not a directory")

    register_coco_instances("training_data", 
                        {}, 
                        path_join(json_dir, "train.json"), 
                        imgs_root) 
    register_coco_instances("val_data", 
                            {}, 
                            path_join(json_dir, "val.json"), 
                            imgs_root)
    register_coco_instances("test_data", 
                            {}, 
                            path_join(json_dir, "test.json"), 
                            imgs_root)