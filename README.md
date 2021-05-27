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

#TODO: note about LabelImg and how to use; show file structure, and mention  Dropbox. 

### The `.yaml` file

The successful execution of a command requires filling out a single `.yaml` configuration file. The configuration file provides the arguments for all of `detectools`' commands. By default, `detectools` will look for a file called **`config.yaml`** in the directory from which you run a `detectools` command. For this reason, I suggest that you name your `.yaml` file  `config.yaml`. Otherwise, you can specify a particular `.yaml` file like so:

```
detectools --config <path/to/config.yaml> <command>
```

For example, if the `.yaml` file you want to use has the path `~/tmp/my_weird_config.yaml`, and you want to run the `undistort` command, you'd input:

```bash
detectools --config ~/tmp/my_weird_config.yaml undistort
```

Each key in the `.yaml` configuration file, except for the `base` key refers to a `detectools` command. The value of each of these keys is a dictionary that specifies the parameters for that `detectools` command. The values of the `base` key are common to a lot of `detectools` commands, and so are factored out. An explanation of the `base` key's values are:

- `root`: 
- `imgs_root`: 

Make sure you do not have any trailing spaces in the `.yaml` file. An example `config.yaml` file is provided in the repository. 

### Commands

The outputs of `detectools`' commands never overwrite existing files, without first asking for user confirmation. `detectools`' commands and their respective `.yaml` file arguments are documented below:

#### `print-config`

This command prints the contents of the `.yaml` configuration file. It does not have any `.yaml` parameters.

#### `voc-to-coco`

This command batch converts Pascal VOC-style `.xml` annotation files to  COCO-style `.json` annotation files. It can be used for converting the annotation outputs of [`labelImg`](https://github.com/tzutalin/labelImg) to the required annotation format for [Detectron2](https://github.com/facebookresearch/detectron2).  Its `.yaml` parameters are :

- `ann_root`:
- `labels`:
- `train_frac`:

This command returns, in the `root` directory, four `.json` files that specify the images and annotations used for each fraction of the split data. The `train.json` contains information for the training data fraction, the `val.json` contains information for the  evaluation data fraction, the `test.json` contains information for the test data fraction, and the `all.json` contains information for the entire dataset. 

#### ` see-data`

This command .... Its `.yaml` parameters are :

- `number_of_imgs`: 
- `scale`:

This command returns a ....

#### `train-model`

This command .... Its `.yaml` parameters are :

- `learning_rate`:
- `lr_decay_policy`:
- `max_iter`:
- `eval_period`:
- `checkpoint_period`:

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