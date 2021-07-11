from pathlib import Path
from os.path import expanduser, join
from pprint import pprint
import subprocess

import click
import yaml


pass_config = click.make_pass_decorator(dict)

# TODO: add a DEFAULT_CONFIG ?

def load_config(fname):
    if fname == None:
        fname = "config.yaml"

    if Path(fname).exists():
        with open(fname) as f:
            config = yaml.safe_load(f) 
    else:
        config = dict()
        exit("You did not pass in a .yaml file! Please pass in a .yaml file.")

    return config

@click.group()
@click.option('--config', type=click.Path(exists=True, dir_okay=False),
              help='The config file to use instead of the default `config.yaml`.')
@click.pass_context
def cli(ctx, config):
    ctx.obj = load_config(config)

@cli.command()
@pass_config
def print_config(config):
    print("")
    pprint(config)
    print("")

@cli.command()
@pass_config
def voc_to_coco(config):
    from detectools import voc_to_coco
    click.echo("\nConverting ...")
    voc_to_coco.main(config)

@cli.command()
@pass_config
def see_data(config):
    from detectools import see_data
    click.echo("\nShowing training data ...")
    see_data.main(config)

@cli.command()
@pass_config
def train_model(config):
    from detectools import train_model
    click.echo("\nTraining model ...")
    click.echo("Run this command with the tee command to save the terminal output. \
               E.g. `detectools train-model|tee my_file.txt")
    train_model.main(config)

@cli.command()
@pass_config
def see_tensorboard(config):
    root = expanduser(config["base"]["root"])
    model_dir = join(root, "outputs")
    subprocess.run(["tensorboard", "--logdir", model_dir])
    # TODO: Have it open http://localhost:6006/ (Press CTRL+C to quit)

@cli.command()
@pass_config
def test_model(config):
    from detectools import test_model
    click.echo("\nTesting model ...")
    test_model.main(config)

@cli.command()
@pass_config
def analyze_vids(config):
    from detectools import analyze_vids
    click.echo("\nAnalyzing videos ...")
    analyze_vids.main(config)


if __name__ == "__main__":
    cli()