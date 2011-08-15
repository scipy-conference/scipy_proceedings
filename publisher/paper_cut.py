"""
Parse pdflatex output for paper count, and store in a .ini file.
"""

import sys
import re
import os

import options

regexp = re.compile('Output written on paper.pdf \((\d+) pages')
cfgname = 'paper_stats.json'

d = options.cfg2dict(cfgname)

for line in sys.stdin:
    m = regexp.match(line)
    if m:
        pages = m.groups()[0]
        d.update({'pages': pages})
        break

options.dict2cfg(d, cfgname)
