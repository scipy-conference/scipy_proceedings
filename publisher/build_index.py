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


def fill_toc_template(template_file, output_file, template):
    with open(template_file, 'r') as f:
        toc = f.read()

    data = ''
    for entry in toc_entries:
        data += (template % entry)

    toc = toc.replace('%(content)s', data)

    with codecs.open(output_file, encoding='utf-8', mode='w') as f:
        f.write(toc)


toc_template_latex = r'''
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

toc_template_html = r'''
<!---------------------------------------------------------->
<p>
<span class="title">%(title)s</span>
<span class="pagenr">%(page)s</span><br/>
<span class="authors">%(authors)s</span>
</p>
<!---------------------------------------------------------->

'''

print "Constructing LaTeX table of contents..."
fill_toc_template(os.path.join(cover_dir, 'toc_template.tex'),
                  os.path.join(output_dir, 'toc.tex'),
                  toc_template_latex)

print "Constructing HTML table of contents..."
fill_toc_template(os.path.join(cover_dir, 'toc_template.html'),
                  os.path.join(output_dir, 'toc.html'),
                  toc_template_html)
