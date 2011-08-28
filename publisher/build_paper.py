#!/usr/bin/env python

import docutils.core as dc
import os.path
import sys
import re
import glob
import shutil

from writer import writer
from conf import papers_dir, output_dir
import options

header = r'''
.. role:: ref

.. role:: label

.. raw::  latex

  \InputIfFileExists{page_numbers.tex}{}{}
  \newcommand*{\docutilsroleref}{\ref}
  \newcommand*{\docutilsrolelabel}{\label}

'''


def rst2tex(in_path, out_path):

    options.mkdir_p(out_path)
    for file in glob.glob(os.path.join(in_path,'*')):
        shutil.copy(file, out_path)

    scipy_style = os.path.join(os.path.dirname(__file__),'_static/scipy.sty')
    shutil.copy(scipy_style, out_path)
    preamble = r'''\usepackage{scipy}'''
    
    # Add the LaTeX commands required by Pygments to do syntax highlighting
    
    try:
        import pygments
    except ImportError:
        import warnings
        warnings.warn(RuntimeWarning('Could not import Pygments. '
                                     'Syntax highlighting will fail.'))
        pygments = None
    
    if pygments:
        from pygments.formatters import LatexFormatter
        from writer.sphinx_highlight import SphinxStyle
    
        preamble += LatexFormatter(style=SphinxStyle).get_style_defs()
    
    
    settings = {'documentclass': 'IEEEtran',
                'use_verbatim_when_possible': True,
                'use_latex_citations': True,
                'latex_preamble': preamble,
                'documentoptions': 'letterpaper,compsoc,twoside'}
    
    
    try:
        rst, = glob.glob(os.path.join(in_path, '*.rst'))
    except ValueError:
        raise RuntimeError("Found more than one input .rst--not sure which one to use.")
    
    content = header + open(rst, 'r').read()
    
    tex = dc.publish_string(source=content, writer=writer,
                            settings_overrides=settings)
    
    stats_file = os.path.join(out_path, 'paper_stats.json')
    d = options.cfg2dict(stats_file)
    d.update(writer.document.stats)
    options.dict2cfg(d, stats_file)
    
    tex_file = os.path.join(out_path, 'paper.tex')
    with open(tex_file, 'w') as f:
        f.write(tex)

def tex2pdf(out_path):

    import shlex, subprocess
    command_line = 'cd '+out_path+' ; pdflatex paper.tex'
    
    run = subprocess.Popen(command_line, shell=True, stdout=subprocess.PIPE)
    out, err = run.communicate()
    
    run = subprocess.Popen(command_line, shell=True, stdout=subprocess.PIPE)
    out, err = run.communicate()
    return out

def page_count(pdflatex_stdout, paper_dir):
   """
   Parse pdflatex output for paper count, and store in a .ini file.
   """

   regexp = re.compile('Output written on paper.pdf \((\d+) pages')
   cfgname = os.path.join(paper_dir,'paper_stats.json')

   d = options.cfg2dict(cfgname)
   
   for line in pdflatex_stdout.splitlines():
       m = regexp.match(line)
       if m:
           pages = m.groups()[0]
           d.update({'pages': int(pages)})
           break
   
   options.dict2cfg(d, cfgname)

def build_paper(paper_id):
   out_path = os.path.join(output_dir, paper_id)
   in_path = os.path.join(papers_dir, paper_id)
   print "Building:", paper_id
   
   rst2tex(in_path, out_path)
   pdflatex_stdout = tex2pdf(out_path)
   page_count(pdflatex_stdout, out_path)

if __name__ == "__main__":

   if len(sys.argv) != 2:
       print "Usage: build_paper.py paper_directory"
       sys.exit(-1)
   
   in_path = sys.argv[1]
   if not os.path.isdir(in_path):
       print("Cannot open directory: %s" % in_path)
       sys.exit(-1)
   
   paper_id = os.path.basename(in_path)
   build_paper(paper_id)
