#!/usr/bin/env python

import os
import glob
import shutil

from conf import bib_dir, template_dir, html_dir, static_dir, pdf_dir
from options import get_config, mkdir_p
from build_template import bib_from_tmpl, html_from_tmpl, from_template

config = get_config()
mkdir_p(bib_dir)
for file in glob.glob(os.path.join(static_dir,'*.css')):
    shutil.copy(file, html_dir)
html_pdfs = os.path.join(html_dir, 'pdfs')
mkdir_p(html_pdfs)
for file in glob.glob(os.path.join(pdf_dir,'*.pdf')):
    shutil.copy(file, html_pdfs)

bib_from_tmpl('proceedings',config,config['proceedings']['citation_key'])

proc_dict = dict(config.items() +
                {'pdf': 'pdfs/proceedings.pdf'}.items() +
                {'bibtex': 'bib/proc-scipy-2010.bib'}.items())
for dest_fn in ['index','organization']:
    html_from_tmpl(dest_fn+'.html', proc_dict, dest_fn)

for article in config['toc']:
    art_dict = dict(config.items() +
                    {'article': article}.items() +
                    {'pdf': 'pdfs/'+article['paper_id']+'.pdf'}.items() +
                    {'bibtex': 'bib/'+article['paper_id']+'.bib'}.items())
    bib_from_tmpl('article', art_dict, article['paper_id'])
    html_from_tmpl('article.html',art_dict, article['paper_id'])
