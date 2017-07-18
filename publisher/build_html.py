#!/usr/bin/env python

import os
import glob
import shutil
from copy import deepcopy

from conf import bib_dir, template_dir, html_dir, static_dir, pdf_dir
from options import get_config, mkdir_p
from build_template import bib_from_tmpl, html_from_tmpl, from_template

config = get_config()
mkdir_p(bib_dir)
for file in glob.glob(os.path.join(static_dir,'*.css')):
    shutil.copy(file, html_dir)
for file in glob.glob(os.path.join(static_dir,'*.js')):
    shutil.copy(file, os.path.join(html_dir,'..'))
html_pdfs = os.path.join(html_dir, 'pdfs')
mkdir_p(html_pdfs)
for file in glob.glob(os.path.join(pdf_dir,'*.pdf')):
    shutil.copy(file, html_pdfs)

citation_key = config['proceedings']['citation_key'] # e.g. proc-scipy-2010

bib_from_tmpl('proceedings', config, citation_key)

proc_dict = deepcopy(config)
proc_dict.update({
    'pdf': 'pdfs/proceedings.pdf',
    'bibtex': 'bib/' + citation_key + '.bib'
    })

for dest_fn in ['index', 'organization', 'students']:
    html_from_tmpl(dest_fn+'.html', proc_dict, dest_fn)

for article in config['toc']:
    art_dict = deepcopy(config)
    art_dict.update({
        'article': article,
        'pdf': 'pdfs/'+article['paper_id']+'.pdf',
        'bibtex': 'bib/'+article['paper_id']+'.bib',
        })
    bib_from_tmpl('article', art_dict, article['paper_id'])
    html_from_tmpl('article.html', art_dict, article['paper_id'])
