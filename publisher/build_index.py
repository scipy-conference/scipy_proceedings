#!/usr/bin/env python

import os
import sys

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

for d in dirs:
    stats = options.cfg2dict(os.path.join(output_dir, d, 'paper_stats.json'))

    # Write page number snippet to be included in the LaTeX output
    if 'pages' in stats:
        pages.append(stats['pages'])
    else:
        pages.append(1)

    cum_pages.append(cum_pages[-1] + pages[-1])
    start = cum_pages[-2]
    stop = cum_pages[-1] - 1

    print '"%s" from p. %s to %s' % (d, start, stop)

    f = open(os.path.join(output_dir, d, 'page_numbers.tex'), 'w')
    f.write('\setcounter{page}{%s}' % start)
    f.close()

    # Build table of contents
    stats.update({'page': {'start': start,
                           'stop': stop}})
    stats.update({'dir': d})
    toc_entries.append(stats)

toc = {'toc': toc_entries}
options.dict2cfg(toc, toc_conf)
