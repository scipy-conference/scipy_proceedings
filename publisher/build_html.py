#!/usr/bin/env python

import os
import glob
import shutil

from conf import bib_dir, template_dir, html_dir, css_file, pdf_dir
from options import get_config, mkdir_p
from build_template import bib_from_tmpl, art_from_tmpl, from_template

config = get_config()
mkdir_p(bib_dir)
shutil.copy(css_file, html_dir)
html_pdfs = os.path.join(html_dir, 'pdfs')
mkdir_p(html_pdfs)
for file in glob.glob(os.path.join(pdf_dir,'*.pdf')):
    shutil.copy(file, html_pdfs)

bib_from_tmpl('proceedings',config,config['proceedings']['citation_key'])

for dest_fn in ['index.html','organization.html']:
    template_fn = os.path.join(template_dir, dest_fn+'.tmpl')
    from_template(template_fn, config, dest_fn)

for article in config['toc']:
    art_dict = dict(config.items() + {'article': article}.items())
    bib_from_tmpl('article', art_dict, article['paper_id'])
    art_from_tmpl(art_dict, article['paper_id'])
