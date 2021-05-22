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

Using `detectools` is simple! From anywehere, type the following in the command line:

```bash
detectools
```

Doing so will bring up the menu of possible options and commands. To execute a command, specify the command of interest after `detectools`. For example, to run the `print-config` command:

```bash
detectools print-config
```

### The `.yaml` file

The successful execution of a command requires filling out a single `.yaml` configuration file. The configuration file provides the arguments for all of `detectools`' commands. By default, `detectools` will look for a file called **`config.yaml`** in the directory from which you run a `detectools` command. For this reason, I suggest that you name your `.yaml` file  `config.yaml`. Otherwise, you can specify a particular `.yaml` file like so:

```
detectools --config <path/to/config.yaml> <command>
```

For example, if the `.yaml` file you want to use has the path `~/tmp/my_weird_config.yaml`, and you want to run the `undistort` command, you'd input:

```bash
detectools --config ~/tmp/my_weird_config.yaml undistort
```

Each key in the `.yaml` configuration file refers to a `detectools` command. The value of each of these keys is a dictionary that specifies the parameters for that `detectools` command. Make sure you do not have any trailing spaces in the `.yaml` file. An example `config.yaml` file is provided in the repository. 

### Commands

The outputs of `detectools`' commands never overwrite existing files, without first asking for user confirmation. `detectools`' commands and their respective `.yaml` file arguments are documented below: