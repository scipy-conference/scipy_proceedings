#!/usr/bin/env python

from conf import bib_dir
from options import get_config, mkdir_p
from build_template import bib_from_tmpl

config = get_config()
mkdir_p(bib_dir)

bib_from_tmpl('proceedings',config,config['proceedings']['citation_key'])

for article in config['toc']:
    art_dict = dict(config.items() + {'article': article}.items())
    bib_from_tmpl('article', art_dict, article['paper_id'])
