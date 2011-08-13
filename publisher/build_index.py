#!/usr/bin/env python

import glob
import os
import sys
import codecs

if not os.path.exists('publisher'):
    raise RuntimeError('Please start this script from the proceedings root.')

sys.path.insert(0, 'publisher')

import options

output_dir = 'output'
cover_dir = 'cover_material'
dirs = [d for d in glob.glob('%s/*' % output_dir) if os.path.isdir(d)]

pages = []
cum_pages = [1]

toc_entries = []

for d in sorted(dirs):
    stats = options.cfg2dict(os.path.join(d, 'paper_stats.cfg'))

    # Write page number snippet to be included in the LaTeX output
    if 'pages' in stats:
        pages.append(int(stats['pages']))
    else:
        pages.append(1)

    cum_pages.append(cum_pages[-1] + pages[-1])

    print '"%s" from p. %s to %s' % (os.path.basename(d), cum_pages[-2],
                                     cum_pages[-1] - 1)

    f = open(os.path.join(d, 'page_numbers.tex'), 'w')
    f.write('\setcounter{page}{%s}' % cum_pages[-2])
    f.close()

    # Build table of contents
    stats.update({'page': cum_pages[-2]})
    toc_entries.append(stats)


print
print "Writing table of contents..."

with open(os.path.join(cover_dir, 'toc_template.tex'), 'r') as f:
    toc = f.read()
toc = toc.replace(r'\end{document}', '')

entry_template = r'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\large{%(title)s}
\hfill
\textbf{%(page)s}
\\
\hspace{1cm}
%(authors)s
\\
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

'''
for entry in toc_entries:
    toc += (entry_template % entry)

toc += r'\end{document}'

toc_filename = os.path.join(output_dir, 'toc.tex')
with codecs.open(toc_filename, encoding='utf-8', mode='w') as f:
    f.write(toc)

