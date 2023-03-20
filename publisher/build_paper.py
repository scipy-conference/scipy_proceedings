#!/usr/bin/env python
from __future__ import print_function, unicode_literals

from copy import deepcopy
import os
import os.path
import sys
import re
import tempfile
import glob
import shutil
import io
import itertools
import subprocess
import yaml

from distutils import dir_util

from conf import papers_dir, output_dir, status_file, static_dir

import options

header = r'''
.. role:: ref

.. role:: label

.. role:: cite(raw)
   :format: latex

.. |---| unicode:: U+2014  .. em dash, trimming surrounding whitespace
    :trim:

.. |--| unicode:: U+2013   .. en dash
    :trim:


'''


BUILD_SYSTEM_FILES = (
    "template.tex",
    "page_numbers.tex",
    "scipy.sty",
    "README.md",
    "status.sty",
    "resolve-references.lua",
    "latex-table.lua"
)


def detect_paper_type(in_path: str) -> str:
    directory_contents = os.listdir(in_path)
    rst_count = 0
    tex_count = 0
    for path in directory_contents:
        path = path.lower()
        if path.endswith(BUILD_SYSTEM_FILES):
            pass
        elif path.endswith('.rst'): rst_count += 1
        elif path.endswith('.tex'): tex_count += 1
    if rst_count and tex_count:
        raise RuntimeError("Saw .rst and .tex files -- paper source unclear")
    if rst_count:
        return 'rst'
    elif tex_count:
        return 'tex'
    else:
        raise RuntimeError("No .rst or .tex files -- paper source unclear")

def prepare_dir(in_path, out_path, start):
    # copy the whole source folder to the build directory
    dir_util.copy_tree(in_path, out_path)
    base_dir = os.path.dirname(__file__)
    # make the page numbers file
    page_number_file = os.path.join(out_path, 'page_numbers.tex')
    with io.open(page_number_file, 'w', encoding='utf-8') as f:
        f.write('\setcounter{page}{%s}' % start)
    # the status style file gets copied separately, since we need to rename it
    out_file = shutil.copy(status_file, out_path)
    os.rename(out_file, os.path.join(out_path, 'status.sty'))
    # then copy the other static files we need into the build dir
    for filename in ['_static/scipy.sty', '_static/template.tex', '_static/resolve-references.lua', '_static/latex-table.lua']:
        scipy_style = os.path.join(base_dir, filename)
        shutil.copy(scipy_style, out_path)


def prepare_metadata(out_path):
    # find the metadata file and preprocess it for pandoc and the stats file
    try:
        metadata_path, = glob.glob(os.path.join(out_path, '*.yaml'))
    except ValueError:
        raise RuntimeError("Found more than one input .yaml--not sure which "
                           "one to use.")
    with io.open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = yaml.safe_load(f)
    metadata = preprocess_metadata(metadata)
    with io.open(metadata_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(metadata, f)
    # update the stats file based on the metadata we read in
    stats_file = os.path.join(out_path, 'paper_stats.json')
    d = options.cfg2dict(stats_file)
    try:
        d.update(metadata)
        options.dict2cfg(d, stats_file)
    except AttributeError:
        print("Error: no paper configuration found")
    return metadata_path


def rst2tex(out_path, metadata_path):
    # find the paper and add custom rst header stuff
    try:
        paper_path, = glob.glob(os.path.join(out_path, '*.rst'))
    except ValueError:
        raise RuntimeError("Found more than one input .rst--not sure which "
                           "one to use.")
    with io.open(paper_path, mode='r', encoding='utf-8') as f:
        content = header + f.read()
    with io.open(paper_path, mode='w', encoding='utf-8') as f:
        f.write(content)
    # call pandoc to generate latex input
    command = [
        'pandoc',
        '-s',
        '-o', 'paper.tex',
        '--metadata-file', metadata_path,
        '--lua-filter', 'resolve-references.lua',
        # '--lua-filter', 'latex-table.lua',
        '--template', 'template.tex',
        paper_path
    ]
    subprocess.run(command, cwd=out_path)


def tex2tex(out_path, metadata_path):
    # find the paper
    paper_paths = glob.glob(os.path.join(out_path, '*.tex'))
    paper_paths = [p for p in paper_paths if not (p.endswith(BUILD_SYSTEM_FILES))]
    if len(paper_paths) > 1:
        raise RuntimeError("Found more than one input .tex--not sure which "
                           "one to use.")
    else:
        paper_path = paper_paths[0]
    # call pandoc to generate latex input
    command = [
        'cp', paper_path, 'paper.tex'
    ]
    subprocess.run(command, cwd=out_path)


def preprocess_metadata(meta):
    new = deepcopy(meta)
    authors = new['authors']
    new['copyright_holder'] = authors[0]['name'] + " et al."
    for a in authors:
        i = a.get('institution')
        if not i:
            a['institution'] = []
        elif isinstance(i, str):
            a ['institution'] = [i]
    new['author'] = [a['name'] for a in authors]
    new['author_email'] = [a['email'] for a in authors]
    new['author_institution'] = list(itertools.chain(a['institution'] for a in authors))
    new['author_institution_map'] = {a['name']: a['institution'].copy() for a in authors if a['institution']}
    new['author_orcid_map'] = {a['name']: a['orcid'] for a in authors if a.get('orcid')}
    institutions = {}
    order = 1
    for a in authors:
        for i in a['institution']:
            if i not in institutions:
                institutions[i] = {'order': order, 'authors': []}
                order += 1
            institutions[i]['authors'].append(a)
    new['institutions'] = [{'name': k, 'order': v['order']} for k, v in institutions.items()]
    for a in authors:
        numbers = [institutions[i]['order'] for i in a['institution']]
        if numbers:
            a['institution'] = numbers
        else:
            a.pop('institution')
    return new


def tex2pdf(out_path):

    # Sometimes Latex want us to rebuild because labels have changed.
    # We will try at most 5 times.
    for i in range(5):
        out, retry = tex2pdf_singlepass(out_path)
        if not retry:
            # Building succeeded or failed outright
            break
    return out


def tex2pdf_singlepass(out_path):
    """
    Returns
    -------
    out : str
        LaTeX output.
    retry : bool
        Whether another round of building is needed.
    """

    command_line = 'pdflatex -halt-on-error paper.tex'

    # -- dummy tempfile is a hacky way to prevent pdflatex
    #    from asking for any missing files via stdin prompts,
    #    which mess up our build process.
    dummy = tempfile.TemporaryFile()

    run = subprocess.Popen(command_line, shell=True,
            stdin=dummy,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=out_path,
            )
    out, err = run.communicate()

    if b"Fatal" in out or run.returncode:
        print("PDFLaTeX error output:")
        print("=" * 80)
        print(out.decode('utf-8'))
        print("=" * 80)
        if err:
            print(err.decode('utf-8'))
            print("=" * 80)

        # Errors, exit early
        return out, False

    # Compile BiBTeX if available
    stats_file = os.path.join(out_path, 'paper_stats.json')
    d = options.cfg2dict(stats_file)
    bib_file = os.path.join(out_path, d["bibliography"] + '.bib')

    if os.path.exists(bib_file):
        bibtex_cmd = 'bibtex paper && ' + command_line
        run = subprocess.Popen(bibtex_cmd, shell=True,
                stdin=dummy,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=out_path,
                )
        out_bib, err = run.communicate()
        if err or b'Error' in out_bib:
            print("Error compiling BiBTeX")
            print("bibtex error output:")
            print("=" * 80)
            print(out_bib)
            print("=" * 80)
            return out_bib, False

    if b"Label(s) may have changed." in out:
        return out, True

    return out, False


def page_count(pdflatex_stdout, paper_dir):
    """
    Parse pdflatex output for paper count, and store in a .ini file.
    """
    if pdflatex_stdout is None:
        print("*** WARNING: PDFLaTeX failed to generate output.")
        return

    regexp = re.compile(b'Output written on paper.pdf \((\d+) pages')
    cfgname = os.path.join(paper_dir, 'paper_stats.json')

    d = options.cfg2dict(cfgname)

    for line in pdflatex_stdout.splitlines():
        m = regexp.match(line)
        if m:
            pages = m.groups()[0]
            d.update({'pages': int(pages)})
            break
    options.dict2cfg(d, cfgname)


def build_paper(paper_id, start=1):
    out_path = os.path.join(output_dir, paper_id)
    in_path = os.path.join(papers_dir, paper_id)
    print("Building:", paper_id)

    # do this early, so we fail before copying anything
    paper_type = detect_paper_type(in_path)

    # copy a bunch of stuff
    options.mkdir_p(out_path)
    prepare_dir(in_path, out_path, start)
    metadata_path = prepare_metadata(out_path)

    if paper_type == 'rst':
        rst2tex(out_path, metadata_path)
    elif paper_type == 'tex':
        tex2tex(out_path, metadata_path)
    pdflatex_stdout = tex2pdf(out_path)
    page_count(pdflatex_stdout, out_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: build_paper.py paper_directory")
        sys.exit(-1)

    in_path = os.path.normpath(sys.argv[1])
    if not os.path.isdir(in_path):
        print("Cannot open directory: %s" % in_path)
        sys.exit(-1)

    paper_id = os.path.basename(in_path)
    build_paper(paper_id)
