from pathlib import Path
from pprint import pprint

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


if __name__ == "__main__":
    cli()