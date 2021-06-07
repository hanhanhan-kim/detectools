# detectools

A CLI wrapper for working with [Detectron2](https://github.com/facebookresearch/detectron2). 

Tested on Ubuntu 18.04. 

## Installation:

1. Clone this repository:

```bash
git clone https://github.com/hanhanhan-kim/detectools
```

2. Install the Anaconda environment from the repo's root directory, where `conda_env.yaml` is housed:

```bash
conda env create -f conda_env.yaml
```

3. Activate the environment:

```bash
conda activate detectools
```

4. Install the `detectools` Python package from the repo's root directory:

```bash
pip install -e .
```

## How to use:

Using `detectools` is simple! From anywhere, type the following in the command line:

```bash
detectools
```

Doing so will bring up the menu of possible options and commands. To execute a command, specify the command of interest after `detectools`. For example, to run the `print-config` command:

```bash
detectools print-config
```

`detectools` assumes that you are starting off with labeled data formatted in Pascal VOC-style `.xml` files. A [lot of data labeling software](https://github.com/heartexlabs/awesome-data-labeling) can output Pascal VOC-style `xml`s. For my own data, I've enjoyed using [`labelImg`](https://github.com/tzutalin/labelImg). When generating labels, make sure to save them in a flat directory. Whereas the raw image data can exist in a nested directory structure, the labeled `xml` files cannot. A sample suitable file structure is depicted below:

```
my_pngs
├── yesterday
|	├── img_00.png
|	├── img_01.png
|	└── bedtime
|		└── img_02.png
└── today
	├── img_03.png
	└── img_04.png

my_xmls
├── pvoc_00.xml
├── pvoc_01.xml
├── pvoc_02.xml
├── pvoc_03.xml
└── pvoc_04.xml
```

Note that failing to save the `.xml` labels in a flat directory, and then moving the `.xml`s to a flat directory afterwards, results in labeling errors. This oversight can be easily addressed in the `labelImg` GUI by selecting `File` → `Save As`, and then saving the `.xml` to the desired flat directory. Note that the use of a flat directory requires that each `.xml` be uniquely named. 

In addition, labels should be generated on the same machine. For example, saving some `.xml`s on one machine, and then saving other `.xml`s on another machine, will result in errors, even if those `.xml`s are saved to a Dropbox folder common to both machines. This issue can be resolved, however, by editing the part of the `.xml` that records the file path. See the commented out [Line 42](https://github.com/hanhanhan-kim/detectools/blob/master/detectools/voc_to_coco.py#L42) in `voc_to_coco.py` as an example. 

A `detectools`-based workflow will look like this:

0. Annotate images with your preferred labeling software, e.g. [`labelImg`](https://github.com/tzutalin/labelImg). 
   - If you must extract images from video data, prior to step 1, see the open-sourced [`vidtools`](https://github.com/hanhanhan-kim/vidtools) toolkit's `vid-to-imgs` command. 

1. Run the `voc-to-coco` command.
2. Run the `see-data` command (optional).
3. Run the `train-model` command.
4. Run the `see-tensorboard` command (optional, but highly recommended). 
5. Run the `eval-model` command. 
6. Run the `analyze-vids` command. 

### The `.yaml` file

The successful execution of a command requires filling out a single `.yaml` configuration file. An example file is provided in the repository. The configuration file provides the arguments for all of `detectools`' commands. By default, `detectools` will look for a file called **`config.yaml`** in the directory from which you run a `detectools` command. For this reason, I suggest that you name your `.yaml` file  `config.yaml`. Otherwise, you can specify a particular `.yaml` file like so:

```
detectools --config <path/to/config.yaml> <command>
```

For example, if the `.yaml` file you want to use has the path `~/tmp/my_weird_config.yaml`, and you want to run the `undistort` command, you'd input:

```bash
detectools --config ~/tmp/my_weird_config.yaml undistort
```

Make sure you do not have any trailing spaces in the `.yaml` file.

Each key in the `.yaml` configuration file refers to a `detectools` command, and the value of each key specifies a parameter for that `detectools` command. The only key in the `.yaml` file that does not refer to a `detectools` command is the `base` key.

#### `base`

The values of the `base` key are common to a lot of `detectools` commands, and so are factored out. The `.yaml` parameters are:

- `root` (string): The root directory for housing `detectools` command outputs. Running the `voc-to-coco` command will output the `jsons` subdirectory at this root directory. Running the `train-model` command will output the `outputs` subdirectory at this root directory. 
- `imgs_root` (string): The root directory that houses the images that were used for generating the labeled data. 

Do not move the contents of `root`.

### Commands

The outputs of `detectools`' commands never overwrite existing files, without first asking for user confirmation (TODO). `detectools`' commands and their respective `.yaml` file arguments are documented below:

#### `print-config`

This command prints the contents of the `.yaml` configuration file. It does not have any `.yaml` parameters.

#### `voc-to-coco`

This command batch converts Pascal VOC-style `.xml` annotation files to  COCO-style `.json` annotation files. It can be used for converting the annotation outputs of [`labelImg`](https://github.com/tzutalin/labelImg) to the required annotation format for [Detectron2](https://github.com/facebookresearch/detectron2).  Its `.yaml` parameters are :

- `ann_root` (string): The root directory that houses the Pascal VOC-style `.xml` files. **Must be a flat directory**.
- `labels` (list of strings): A list of the labels found in the labeled data. 
- `train_frac` (float): The fraction of the labeled data to be used for training the model. The remaining data is evenly split between the evaluation and test fractions. 

This command returns, in the `root` directory, four `.json` files that specify the images and annotations used for each fraction of the split data. The `train.json` contains information for the training data fraction, the `val.json` contains information for the  evaluation data fraction, the `test.json` contains information for the test data fraction, and the `all.json` contains information for the entire dataset. 

#### ` see-data`

This command shows the labeled images in the training data fraction. Its `.yaml` parameters are:

- `number_of_imgs` (integer): The number of randomly sampled images to show. 
- `scale` (float): The factor by which to scale the displayed image. A scale of `1.0` will display the true size of the image. 

This command returns nothing. It just shows a random sample of labeled images from the training data. 

#### `train-model`

This command trains the [**Faster R-CNN**](https://arxiv.org/abs/1506.01497) Detectron2 model. It does not support other object detection algorithms that are supported in Detectron2, such as Mask R-CNN and RetinaNet. Its `.yaml` parameters are : 

- `learning_rate` (float): The model's initial [learning rate](https://en.wikipedia.org/wiki/Learning_rate), i.e. the hyperparameter that determines the degree to which to adjust the model in response to the estimated error. A reasonable default value is 0.2. 
- `lr_decay_policy` (list of integers): 
- `max_iter` (integer):
- `eval_period` (integer):
- `checkpoint_period` (integer): 

This command returns a ....

#### `see-tensorboard`

This command displays the [TensorBoard](https://www.tensorflow.org/tensorboard) for the trained model at its latest iteration. It does not have any `.yaml` parameters.

#### `eval-model`

This command .... Its `.yaml` parameters are :

- `scale`:
- `do_show`:

This command returns a ....

#### `analyze-vids`

This command .... Its `.yaml` parameters are :

- `model_pth`:
- `score_cutoff`:
- `vids_root`:
- `frame_rate`: 

This command returns a ....