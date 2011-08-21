#!/usr/bin/env python

import os
import sys
import codecs

if not os.path.exists('publisher'):
    raise RuntimeError('Please start this script from the proceedings root.')

sys.path.insert(0, 'publisher')

import conf
import options

output_dir = conf.output_dir
cover_dir = conf.cover_dir
dirs = conf.dirs

pages = []
cum_pages = [1]

toc_entries = []

for d in dirs:
    stats = options.cfg2dict(os.path.join(output_dir, d, 'paper_stats.json'))

    # Write page number snippet to be included in the LaTeX output
    if 'pages' in stats:
        pages.append(int(stats['pages']))
    else:
        pages.append(1)

    cum_pages.append(cum_pages[-1] + pages[-1])

    print '"%s" from p. %s to %s' % (d, cum_pages[-2],
                                     cum_pages[-1] - 1)

    f = open(os.path.join(output_dir, d, 'page_numbers.tex'), 'w')
    f.write('\setcounter{page}{%s}' % cum_pages[-2])
    f.close()

    # Build table of contents
    stats.update({'page': cum_pages[-2]})
    stats.update({'dir': d})
    toc_entries.append(stats)

toc = {'toc': toc_entries}
options.dict2cfg(toc, os.path.join(output_dir, 'toc.json'))
