#!/usr/bin/env python

import os
import sys
import shutil
import subprocess

import conf
import options
from build_paper import build_paper

output_dir = conf.output_dir
build_dir  = conf.build_dir
bib_dir    = conf.bib_dir
pdf_dir    = conf.pdf_dir
toc_conf   = conf.toc_conf
proc_conf  = conf.proc_conf
dirs       = conf.dirs


def paper_stats(paper_id, start):
    stats = options.cfg2dict(os.path.join(output_dir, paper_id, 'paper_stats.json'))

    # Write page number snippet to be included in the LaTeX output
    if 'pages' in stats:
        pages = stats['pages']
    else:
        pages = 1

    stop = start + pages - 1

    print('"%s" from p. %s to %s' % (paper_id, start, stop))

    with open(os.path.join(output_dir, paper_id, 'page_numbers.tex'), 'w') as f:
        f.write('\setcounter{page}{%s}' % start)

    # Build table of contents
    stats.update({'page': {'start': start,
                           'stop': stop}})
    stats.update({'paper_id': paper_id})

    return stats, stop

if __name__ == "__main__":

    start = 0
    toc_entries = []

    options.mkdir_p(pdf_dir)
    for paper_id in dirs:
        build_paper(paper_id)

        stats, start = paper_stats(paper_id, start + 1)
        toc_entries.append(stats)

        build_paper(paper_id)

        src_pdf = os.path.join(output_dir, paper_id, 'paper.pdf')
        dest_pdf = os.path.join(pdf_dir, paper_id+'.pdf')
        shutil.copy(src_pdf, dest_pdf)

        command_line = 'cd '+pdf_dir+' ; pdfannotextractor '+paper_id+'.pdf'
        run = subprocess.Popen(command_line, shell=True, stdout=subprocess.PIPE)
        out, err = run.communicate()

    toc = {'toc': toc_entries}
    options.dict2cfg(toc, toc_conf)
