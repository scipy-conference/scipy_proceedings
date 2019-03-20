#!/usr/bin/env python
from __future__ import print_function, unicode_literals

import docutils.core as dc
import os
import os.path
import sys
import re
import tempfile
import glob
import shutil
import io

from distutils import dir_util

from writer import writer
from conf import papers_dir, output_dir, status_file, static_dir

import options

header = r'''
.. role:: ref

.. role:: label

.. role:: cite(raw)
   :format: latex

.. raw::  latex

    \InputIfFileExists{page_numbers.tex}{}{}
    \newcommand*{\docutilsroleref}{\ref}
    \newcommand*{\docutilsrolelabel}{\label}
    \newcommand*\DUrolecode[1]{#1}
    \providecommand*\DUrolecite[1]{\cite{#1}}

.. |---| unicode:: U+2014  .. em dash, trimming surrounding whitespace
    :trim:

.. |--| unicode:: U+2013   .. en dash
    :trim:


'''

def rst2tex(in_path, out_path):

    dir_util.copy_tree(in_path, out_path)

    base_dir = os.path.dirname(__file__)
    out_file = shutil.copy(status_file, out_path)
    os.rename(out_file, os.path.join(out_path, 'status.sty'))
    scipy_style = os.path.join(base_dir, '_static/scipy.sty')
    shutil.copy(scipy_style, out_path)
    preamble = u'''\\usepackage{scipy}'''

    # Add the LaTeX commands required by Pygments to do syntax highlighting

    pygments = None

    try:
        import pygments
    except ImportError:
        import warnings
        warnings.warn(RuntimeWarning('Could not import Pygments. '
                                     'Syntax highlighting will fail.'))

    if pygments:
        from pygments.formatters import LatexFormatter
        from writer.sphinx_highlight import SphinxStyle

        preamble += LatexFormatter(style=SphinxStyle).get_style_defs()

    settings = {'documentclass': 'IEEEtran',
                'use_verbatim_when_possible': True,
                'use_latex_citations': True,
                'latex_preamble': preamble,
                'documentoptions': 'letterpaper,compsoc,twoside',
                'halt_level': 3,  # 2: warn; 3: error; 4: severe
                }

    try:
        rst, = glob.glob(os.path.join(in_path, '*.rst'))
    except ValueError:
        raise RuntimeError("Found more than one input .rst--not sure which "
                           "one to use.")

    with io.open(rst, mode='r', encoding='utf-8') as f:
        content = header + f.read()
    
    tex = dc.publish_string(source=content, writer=writer,
                            settings_overrides=settings)

    stats_file = os.path.join(out_path, 'paper_stats.json')
    d = options.cfg2dict(stats_file)
    try:
        d.update(writer.document.stats)
        options.dict2cfg(d, stats_file)
    except AttributeError:
        print("Error: no paper configuration found")

    tex_file = os.path.join(out_path, 'paper.tex')
    with io.open(tex_file, mode='wb') as f:
        try:
            tex = tex.encode('utf-8')
        except (AttributeError, UnicodeDecodeError):
            pass
        f.write(tex)


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

    import subprocess
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
    bib_file = os.path.join(out_path, d.get("bibliography", "nobib") + '.bib')

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
    
    
    options.mkdir_p(out_path)
    page_number_file = os.path.join(out_path, 'page_numbers.tex')
    with io.open(page_number_file, 'w', encoding='utf-8') as f:
        f.write('\setcounter{page}{%s}' % start)

    print('in_path is {}'.format(in_path))
    print('out_path is {}'.format(out_path))

    rstfiles = glob.glob(os.path.join(in_path, '*rst'))
    if len(rstfiles) > 1:
        raise RuntimeError("Found more than one input .rst--not sure which "
                           "one to use.")
    elif len(rstfiles) == 0:
        texfiles = glob.glob(os.path.join(in_path, '*tex'))
        if len(texfiles) == 0:
            raise RuntimeError("Could not find a .rst or .tex file in {}"
                               .format(in_path))
        elif len(texfiles) == 1:
            base_dir = os.path.dirname(__file__)
            dir_util.copy_tree(in_path, out_path)
            shutil.copy(texfiles[0], os.path.join(out_path, 'paper.tex'))
            out_file = shutil.copy(status_file, out_path)
            os.rename(out_file, os.path.join(out_path, 'status.sty'))
            scipy_style = os.path.join(base_dir, '_static/scipy.sty')
            shutil.copy(scipy_style, os.path.join(out_path, 'scipy.sty'))
            stats_file = os.path.join(out_path, 'paper_stats.json')
            d = options.cfg2dict(stats_file)
            try:
                d.update(writer.document.stats)
                options.dict2cfg(d, stats_file)
            except AttributeError:
                print("Error: no paper configuration found")

        else:
            raise RuntimeError("Found no .rst files and more than one .tex "
                               "file in input directory.")

    else:
        rst2tex(in_path, out_path)

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
