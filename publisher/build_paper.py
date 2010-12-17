#!/usr/bin/env python

import docutils.core as dc

from writer import writer

import os.path
import sys
import glob

preamble = r'''
% These preamble commands are from build_paper.py

% PDF Standard Fonts
\usepackage{mathptmx}
\usepackage[scaled=.90]{helvet}
\usepackage{courier}

% Make verbatim environment smaller
\makeatletter
\g@addto@macro\@verbatim\footnotesize
\makeatother

% Do not indent code sections
\renewcommand{\quote}{}

% Provide AMS mathematical commands such as "align"
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{bm}

% Define colours for hyperref
\usepackage{color}

\definecolor{orange}{cmyk}{0,0.4,0.8,0.2}
\definecolor{darkorange}{rgb}{.71,0.21,0.01}
\definecolor{darkblue}{rgb}{.01,0.21,0.71}
\definecolor{darkgreen}{rgb}{.1,.52,.09}

\usepackage{hyperref}
\hypersetup{pdftex,  % needed for pdflatex
  breaklinks=true,  % so long urls are correctly broken across lines
  colorlinks=true,
  urlcolor=blue,
  linkcolor=darkblue,
  citecolor=darkgreen,
  }

% Include graphics for authors who raw-inlined figures
% (then docutils won't automatically add the package)
\usepackage{graphicx}

\ifthenelse{\isundefined{\longtable}}{}{
  \renewenvironment{longtable}{\begin{table}\begin{tabular}}%
    {\end{tabular}\vspace{2mm}\end{table}}
}
'''

settings = {'documentclass': 'IEEEtran',
            'use_verbatim_when_possible': True,
            'use_latex_citations': True,
            'latex_preamble': preamble,
            'documentoptions': 'letterpaper,compsoc,twoside'}

if len(sys.argv) != 3:
    print "Usage: build_paper.py paper_directory target_directory"
    sys.exit(-1)

in_path, out_path = sys.argv[1:]
for p in (in_path, out_path):
    if not os.path.isdir(p):
        print("Cannot open directory: %s" % p)
        sys.exit(-1)

print "Building:", in_path

rst = glob.glob(os.path.join(in_path, '*.rst'))[0]

content = open(rst, 'r').read()
content = '''
.. role:: math(raw)
   :format: latex

''' + content

tex = dc.publish_string(source=content, writer=writer,
                        settings_overrides=settings)

out = open(os.path.join(out_path, 'paper.tex'), 'w')
out.write(tex)
out.close()
