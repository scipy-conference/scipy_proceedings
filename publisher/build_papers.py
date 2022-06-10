#!/usr/bin/env python
from __future__ import unicode_literals

import os
import sys
import shutil
import subprocess
import io

import conf
import options
from build_paper import build_paper
from xreftools import XrefMeta
from doitools import make_doi, make_series_doi

output_dir = conf.output_dir
build_dir  = conf.build_dir
bib_dir    = conf.bib_dir
pdf_dir    = conf.pdf_dir
toc_conf   = conf.toc_conf
proc_conf  = conf.proc_conf
dirs       = conf.dirs
other_dirs = conf.other_dirs
xref_conf = conf.xref_conf
papers_dir = conf.papers_dir
other_conf = conf.other_conf
is_final = conf.status_file_base == 'ready'



def paper_stats(paper_id, start, doi_prefix=None):
    """Pull in stats of paper, return stats and the next paper
    """
    stats = options.cfg2dict(os.path.join(output_dir, paper_id, 'paper_stats.json'))

    pages = stats.get('pages', 1)
    stop = start + pages - 1
    paper_doi = make_doi(doi_prefix)

    print('"%s" from p. %s to %s' % (paper_id, start, stop))

    # Build table of contents
    stats.update({'page': {'start': start,
                           'stop': stop},
                  'paper_id': paper_id,
                  'doi': paper_doi if is_final else ''
                 })

    return stats

def other_stats(track_dir, _id, doi_prefix=None):
    """Pull in stats of slides, return stats and the next paper
    """

    stats = options.cfg2dict(os.path.join(track_dir, _id, 'info.json'))
    doi = make_doi(doi_prefix)

    print('"%s" other entry' % (_id))

    # Build table of contents
    stats.update({
                  'slide_id': _id,
                  'doi': doi if is_final else ''
                })

    return stats


if __name__ == "__main__":

    start = 1
    toc_entries = []
    other_entries = {}

    options.mkdir_p(pdf_dir)
    basedir = os.path.join(os.path.dirname(__file__), '..')
    # load metadata
    scipy_entry = options.cfg2dict(proc_conf)
    doi_prefix = scipy_entry["proceedings"]["xref"]["prefix"]
    issn = scipy_entry['series']['xref']['issn']

    for paper_id in dirs:
        with options.temp_cd(basedir):
            build_paper(paper_id, start=start)

        stats = paper_stats(paper_id, start, doi_prefix)
        start = stats.get('page',{}).get('stop', start) + 1
        toc_entries.append(stats)

        src_pdf = os.path.join(output_dir, paper_id, 'paper.pdf')
        dest_pdf = os.path.join(pdf_dir, paper_id+'.pdf')
        shutil.copy(src_pdf, dest_pdf)

    for track_dir, folder_ids in sorted(other_dirs.items()):
        track = os.path.split(track_dir)[-1]
        other_entries[track] = []
        for folder in folder_ids:
            stats = other_stats(track_dir, folder, doi_prefix)
            other_entries[track].append(stats)


    # load completed TOC
    toc = {'toc': toc_entries}
    # make doi for this year's proceedings and for whole conference (static)
    scipy_entry['proceedings']['doi'] = make_doi(doi_prefix) if is_final else ''
    scipy_entry['series']['doi'] = make_series_doi(doi_prefix, issn)

    # persist metadata
    options.dict2cfg(toc, toc_conf)
    options.dict2cfg(other_entries, other_conf)
    options.dict2cfg(scipy_entry, proc_conf)

    # make crossref submission file
    xref = XrefMeta(scipy_entry, toc_entries, other_entries)
    xref.make_metadata()
    xref.write_metadata(xref_conf)
