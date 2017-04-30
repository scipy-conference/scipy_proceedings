#!/usr/bin/env python
from __future__ import print_function

import docutils.core as dc
import os.path
import sys
import re
import tempfile
import glob
import shutil
import io

from distutils import dir_util

from writer import writer
from conf import papers_dir, output_dir

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
    \providecommand*\DUrolecite[1]{\cite{#1}}

.. |---| unicode:: U+2014  .. em dash, trimming surrounding whitespace
    :trim:

.. |--| unicode:: U+2013   .. en dash
    :trim:


'''

def rst2tex(in_path, out_path):

    dir_util.copy_tree(in_path, out_path)

    base_dir = os.path.dirname(__file__)
    scipy_status = os.path.join(base_dir, '_static/status.sty')
    shutil.copy(scipy_status, out_path)
    scipy_style = os.path.join(base_dir, '_static/scipy.sty')
    shutil.copy(scipy_style, out_path)
    preamble = r'''\usepackage{scipy}'''

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

    rst = _glob_for_one_file(in_path, '*.rst')    

    with io.open(rst, mode='r') as f:
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
        print(out)
        print("=" * 80)
        if err:
            print(err)
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


def _glob_for_one_file(path, pattern):
    try:
        file_found, = glob.glob(os.path.join(path, pattern))
    except ValueError:
        raise RuntimeError("Found more than one input matching {}--not sure which "
                           "one to use.".format(pattern))

    return file_found

class NotebookConverter(object):
    
    def __init__(self, config=None, paper_id='', keep_rst=False, debug=False):
        
        self.paper_id = paper_id
        self.in_path = os.path.join(papers_dir, self.paper_id)
        self.out_path = os.path.join(output_dir, self.paper_id)
        self.keep_rst = keep_rst
        self.debug = debug
        
        if config is None:
            self.config = {}
        else:
            self.config = config

        if self.debug:
            self.debug_dir = os.path.join(papers_dir,'debug')
            try: 
                os.makedirs(self.debug_dir)
            except FileExistsError:
                rst_files = glob.glob(os.path.join(self.debug_dir, "**/*.rst"))
                self.num_debug_files = len(rst_files)
        try:
            self.ipynb_path = _glob_for_one_file(self.in_path, '*.ipynb')
        except RuntimeError:
            self.ipynb_path = None

    def nb_to_rst(self):
        """
        This converts the notebook found on init (at `self.ipynb_path`) to an
        rst file.
        """
        import nbformat
        
        from traitlets.config import Config

        from nbconvert import RSTExporter
        from nbconvert.writers import FilesWriter
        
        with io.open(self.ipynb_path, mode="r") as f:
            nb = nbformat.read(f, as_version=4)
        
        c = Config()
        c.update(self.config)

        rst_exporter = RSTExporter(config = c)
        nbconvert_writer = FilesWriter(build_directory=self.in_path)
        output, resources = rst_exporter.from_notebook_node(nb)
        nbconvert_writer.write(output, resources, notebook_name=self.paper_id)
        self.input_rst_file_path = _glob_for_one_file(self.in_path, '*.rst') 

    def convert(self):
        """
        This executs the `nb_to_rst` conversion step.
        """
        if self.ipynb_path:
            print("Converting {0}.ipynb to {0}.rst".format(self.paper_id))
            self.nb_to_rst()
    
    def create_debug_file_path(self):
        file_basename = os.path.basename(self.input_rst_file_path)
        new_file_name = (os.path.splitext(file_basename)[0] +
                        str(self.num_debug_files + 1) + 
                        os.path.splitext(file_basename)[1])
        self.debug_file_path = os.path.join(self.debug_dir,new_file_name)
        

    def cleanup(self):
        """Applies various cleanup methods for rst converted from a notebook"""

        if self.ipynb_path and not self.keep_rst:
            
            if self.debug:
                self.create_debug_file_path()
                shutil.copy(self.input_rst_file_path, self.debug_file_path())

            os.remove(self.input_rst_file_path)



def build_paper(paper_id):
    """
    Build the paper given the basename of the paper's directory.
    
    E.g., if `papers/00_bibderwalt`, `paper_id = '00_bibderwalt'`.
    
    Input: paper_id: string
    """

    out_path = os.path.join(output_dir, paper_id)
    in_path = os.path.join(papers_dir, paper_id)
    nbconverter = NotebookConverter(paper_id = paper_id) 
    nbconverter.convert()
    print("Building:", paper_id)
    

    rst2tex(in_path, out_path)
    pdflatex_stdout = tex2pdf(out_path)
    page_count(pdflatex_stdout, out_path)
    nbconverter.cleanup()

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
