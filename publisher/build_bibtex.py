#!/usr/bin/env python

import os
import sys
from string import Template
import codecs

from conf import bib_dir
from options import get_config, mkdir_p

proceedingsT = Template('''@Proceedings{${citation_key},
  title     = {${booktitle}},
  booktitle = {${booktitle}},
  year      = {${year}},
  editor    = {${editor}},
  isbn      = {${isbn}}
}''')

inproceedingsT = Template('''@InProceedings{${citation_key},
  author    = {${author}},
  title     = {${title}},
  booktitle = {${booktitle}},
  pages     = {${pages}},
  address   = {${address}},
  year      = {${year}},
  editor    = {${editor}},
}''')

proceedings = get_config()['proceedings']
toc = get_config()['toc']

proc_vals = {
 'citation_key': proceedings['citation_key'],
 'booktitle': proceedings['title']['full'],
 'year': proceedings['year'],
 'editor': ' and '.join(proceedings['editor']),
 'isbn': proceedings['isbn']
}

def bib_write(content, filename, encoding='utf-8', mode='w'):
    with codecs.open(filename, encoding=encoding, mode=mode) as f:
        f.write(content)

mkdir_p(bib_dir)

bib_write(proceedingsT.safe_substitute(proc_vals),
          os.path.join(bib_dir,proc_vals['citation_key']+'.bib'))

for article in toc:
  art_vals = {
    'citation_key': '-'.join([article['dir'],
                                  proceedings['citation_key']]),
    'author': ' and '.join(article['author']),
    'title': article['title'],
    'booktitle': proc_vals['booktitle'],
    'pages': ' - '.join([str(article['page']['start']), str(article['page']['stop'])]),
    'year': proc_vals['year'],
    'editor': proc_vals['editor']
  }
  
  bib_write(inproceedingsT.safe_substitute(art_vals),
          os.path.join(bib_dir,art_vals['citation_key']+'.bib'))
