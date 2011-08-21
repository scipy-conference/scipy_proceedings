#!/usr/bin/env python

import os
import sys
from string import Template
import codecs

import conf
import options

output_dir = conf.output_dir
build_dir  = conf.build_dir
bib_dir    = conf.bib_dir
toc_conf   = conf.toc_conf
proc_conf  = conf.proc_conf
dirs       = conf.dirs

pages = []
cum_pages = [1]

toc_entries = []

proceedings = Template('''@Proceedings{${citation_key},
  title     = {${booktitle}},
  booktitle = {${booktitle}},
  year      = {${year}},
  editor    = {${editor}},
  isbn      = {${isbn}}
}''')

inproceedings = Template('''@InProceedings{${citation_key},
  author    = {${author}},
  title     = {${title}},
  booktitle = {${booktitle}},
  pages     = {${pages}},
  address   = {${address}},
  year      = {${year}},
  editor    = {${editor}},
}''')

proceedings_info = options.cfg2dict(proc_conf)['proceedings']
toc_info = options.cfg2dict(toc_conf)['toc']

proc_vals = {
 'citation_key': proceedings_info['citation_key'],
 'booktitle': proceedings_info['title']['full'],
 'year': proceedings_info['year'],
 'editor': ' and '.join(proceedings_info['editor']),
 'isbn': proceedings_info['isbn']
}

def bib_write(content, filename, encoding='utf-8', mode='w'):
    with codecs.open(filename, encoding=encoding, mode=mode) as f:
        f.write(content)


def mkdir_p(dir):
    if os.path.isdir(dir):
        return
    os.makedirs(dir)

mkdir_p(bib_dir)

bib_write(proceedings.safe_substitute(proc_vals),
          os.path.join(bib_dir,proc_vals['citation_key']+'.bib'))

for article in toc_info:
  art_vals = {
    'citation_key': '-'.join([article['dir'],
                                  proceedings_info['citation_key']]),
    'author': ' and '.join(article['author']),
    'title': article['title'],
    'booktitle': proc_vals['booktitle'],
    'pages': ' - '.join([str(article['page']['start']), str(article['page']['stop'])]),
    'year': proc_vals['year'],
    'editor': proc_vals['editor']
  }
  
  bib_write(inproceedings.safe_substitute(art_vals),
          os.path.join(bib_dir,art_vals['citation_key']+'.bib'))
