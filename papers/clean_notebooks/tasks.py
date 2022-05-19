from pathlib import Path
import shutil

from jinja2 import Template
from invoke import task
import jupytext

_TARGET = Path('~', 'dev', 'ploomber').expanduser()


@task
def setup(c, from_lock=False):
    """Create conda environment
    """
    if from_lock:
        c.run('conda env create --file environment.yml --force')
    else:
        c.run('conda env create --file environment.dev.yml --force')
        c.run('conda env export --no-build --file environment.yml'
              ' --name ploomber-workshop')


@task
def convert(c):
    """Generate README.md and index.md. Convert index.md to index.ipynb
    """
    print('Generating index.ipynb...')
    nb = jupytext.read('index.md')
    jupytext.write(nb, 'index.ipynb')

    files_to_copy = [
        ('_static/workshop.svg', None),
        ('README.md', 'workshop.md'),
    ]

    for f, target in files_to_copy:
        if target is None:
            target = _TARGET / f
        else:
            target = _TARGET / target

        print(f'Copying {f} to {target}...')
        shutil.copy(f, target)


@task
def clear(c):
    """Clear outputs generated when running the code
    """
    playground = Path('playground')

    if playground.exists():
        shutil.rmtree('playground')

    playground.mkdir(exist_ok=True)
    shutil.copy('_sample/nb.ipynb', 'playground/nb.ipynb')
