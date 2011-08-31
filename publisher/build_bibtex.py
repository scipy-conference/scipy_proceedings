#!/usr/bin/env python

import os
import sys
import codecs
import shlex, subprocess

from conf import bib_dir, template_dir
from options import get_config, mkdir_p
from build_template import from_template

config = get_config()
mkdir_p(bib_dir)

def bib_from_tmpl(bib_type, config, target):
    bib_tmpl = os.path.join(template_dir, bib_type + '.bib.tmpl')
    dest_path = os.path.join(bib_dir, target + '.bib')
    from_template(bib_tmpl, config, dest_path)
    command_line = 'recode -d u8..ltex ' + dest_path
    run = subprocess.Popen(command_line, shell=True, stdout=subprocess.PIPE)
    out, err = run.communicate()

bib_from_tmpl('proceedings',config,config['proceedings']['citation_key'])

for article in config['toc']:
    art_dict = dict(config.items() + {'article': article}.items())
    bib_from_tmpl('article', art_dict, article['paper_id'])
