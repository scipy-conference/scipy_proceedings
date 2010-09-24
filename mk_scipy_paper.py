#!/usr/bin/env python
# encoding: utf-8

import os
import re
import sys
import shutil
import codecs
from glob import glob

from docutils import core as docCore

conf_name = 'SciPy2010'

current_dir = os.path.dirname(__file__)
if current_dir == '':
    current_dir = '.'

outdir = current_dir + os.sep + 'output'

sourcedir = current_dir + os.sep + 'source'
try:
    os.mkdir(outdir)
except:
    pass

outfilename = outdir + os.sep + 'booklet.tex'

##############################################################################
# Routines for supervised execution
##############################################################################

from threading import Thread
import os
import signal
from subprocess import Popen
from time import sleep

def delayed_kill(pid, delay=10):
    sleep(delay)
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        pass

def supervise_popen(command, timeout=10):
    process = Popen(command)
    Thread(target=delayed_kill, args=(process.pid,timeout),).start()

    process.wait()



##############################################################################
# LaTeX generation functions.
##############################################################################

def protect(string):
    r''' Protects all the "\" in a string by adding a second one before

    >>> protect(r'\foo \*')
    '\\\\foo \\\\*'
    '''
    return re.sub(r"\\", r"\\\\", string)


def safe_unlink(filename):
    """ Remove a file from the disk only if it exists, if not r=fails silently
    """
    if os.path.exists(filename):
        os.unlink(filename)

rxcountpages = re.compile(r"$\s*/Type\s*/Page[/\s]", re.MULTILINE|re.DOTALL)

def count_pages(filename):
    data = file(filename,"rb").read()
    return len(rxcountpages.findall(data))


def tex2pdf(filename, remove_tex=True, timeout=10, runs=2):
    """ Compiles a TeX file with pdfLaTeX (or LaTeX, if or dvi ps requested)
        and cleans up the mess afterwards
    """
    current_dir = os.getcwd()
    os.chdir(outdir)
    print >> sys.stderr, "Compiling document to pdf"
    basename = os.path.join(outdir, os.path.splitext(os.path.basename(filename))[0])
    if os.path.exists(basename + '.pdf'):
        os.unlink(basename + '.pdf')
    for _ in range(runs):
        supervise_popen(("pdflatex",  "--interaction", "scrollmode",
                        os.path.basename(filename)), timeout=timeout)
    error_file = None
    errors =  file(os.path.abspath('../' + basename + '.log')).readlines()[-1]
    if not os.path.exists(basename + '.pdf') or \
                                    "Fatal error" in errors:
        error_file = os.path.abspath(basename + '.log')
    if remove_tex:
        safe_unlink(filename+".tex")
        safe_unlink(filename+".log")
    safe_unlink(filename+".aux")
    safe_unlink(filename+".out")
    os.chdir(current_dir)
    return error_file


def rst2latex(rst_string, no_preamble=True, allow_latex=True):
    """ Calls docutils' engine to convert a rst string to a LaTeX file.
    """
    overrides = {'output_encoding': 'utf-8', 'initial_header_level': 3,
                 'no_doc_title': True, 'use_latex_citations': True, 
                 'use_latex_footnotes':True}
    if allow_latex:
        rst_string = u'''.. role:: math(raw)
                    :format: latex
                    \n\n''' + rst_string
    tex_string = docCore.publish_string(
                source=rst_string,
                writer_name='latex2e', 
                settings_overrides=overrides)
    if no_preamble:
        extract_document = \
            re.compile(r'.*\\begin\{document\}(.*)\\end\{document\}',
            re.DOTALL)
        matches = extract_document.match(tex_string)
        tex_string = matches.groups()[0]
    return tex_string


def get_latex_preamble():
    """ Retrieve the required preamble from docutils.
    """
    full_document = rst2latex('\n', no_preamble=False)
    preamble = re.split(r'\\begin\{document\}', full_document)[0]
    ## Remove the documentclass.
    preamble = r"""
                %s
                \makeatletter
                \newcommand{\class@name}{gael}
                \makeatother
                \usepackage{ltxgrid}
                %s
                """ % (
                    preamble.split('\n')[0],
                    '\n'.join(preamble.split('\n')[1:]),
                )
    return preamble


##############################################################################
# Functions to generate part of the booklet
##############################################################################
def addfile(outfile, texfilename):
    """ Includes the content of a tex file in our outfile.
    """
    include = codecs.open(texfilename, 'r')
    data = include.readlines()
    outfile.write(ur'\thispagestyle{empty}' + u'\n')
    outfile.writelines(data)


def preamble(outfile):
    outfile.write(r'''
    %s
    \usepackage{abstracts}
    \usepackage{ltxgrid}
    \usepackage{amssymb,latexsym,amsmath,amsthm}
    \usepackage{longtable}
    \geometry{left=.8cm, textwidth=17cm, bindingoffset=0.6cm,
                textheight=25.3cm, twoside}
    \usepackage{hyperref}
    \hypersetup{pdftitle={Proceedings of the 8th Annual Python in Science Conference}}
    \begin{document}

    '''.encode('utf-8') % get_latex_preamble())

    # XXX SciPy08 should not be hard coded, but to run out of the webapp

def hack_include_graphics(latex_text):
    """ Replaces all the \includegraphics call with call that impose the
        width to be 0.9\linewidth.
    """
    latex_text = re.sub(r'\\setlength\{\\rightmargin\}\{\\leftmargin\}',
                        r'\\setlength{\\leftmargin}{4ex}\\setlength{\\rightmargin}{0ex}',
                        latex_text)
    latex_text = re.sub(r'\\begin\{quote\}\n\\begin\{itemize\}',
                        r'\\begin{itemize}',
                        latex_text)
    latex_text = re.sub(r'\\end\{itemize\}\n\\end\{quote\}',
                        r'\\end{itemize}',
                        latex_text)
    latex_text = re.sub(r'\\includegraphics(\[.*\])?\{',
                        r'\includegraphics[width=\linewidth]{',
                        latex_text)
    latex_text = re.sub(r'\\href\{([^}]+)\}\{http://(([^{}]|(\{[^}]*\}))+)\}', 
                        r'''%
% Break penalties to have URL break easily:
\mathchardef\UrlBreakPenalty=0
\mathchardef\UrlBigBreakPenalty=0
%\hskip 0pt plus 2em
\href{\1}{\url{\1}}''',
                        latex_text)
    latex_text = re.sub(r'\\href\{([^}]+)\}\{https://(([^{}]|(\{[^}]*\}))+)\}', 
                        r'''%
% Break penalties to have URL break easily:
\mathchardef\UrlBreakPenalty=0
\mathchardef\UrlBigBreakPenalty=0
\linebreak
\href{\1}{\url{\1}}''',
                        latex_text)

    return latex_text


def render_abstract(outfile, abstract, start_page=None):
    """ Writes the LaTeX string corresponding to one abstract.
    """
    if start_page is not None:
        outfile.write(r"""
\setcounter{page}{%i}
""" % start_page)
    else:
        if hasattr(abstract, 'start_page'):
            start_page = abstract.start_page
        else:
            start_page = 1
    if not abstract.authors:
        author_list = abstract.owners
    else:
        author_list = abstract.authors
    authors = []
    for author in author_list:
        # If the author has no surname, he is not an author 
        if author.surname:
            if author.email_address:
                email = r'(\email{%s})' % author.email_address
            else:
                email = ''
            authors.append(ur'''\otherauthors{
                            %s %s
                            %s --
                            \address{%s, %s \sc{%s}}
                            }''' % (author.first_names, author.surname,
                                    email,
                                    author.institution,
                                    author.address,
                                    author.country))
    if authors:
        authors = u'\n'.join(authors)
        authors += r'\addauthorstoc{%s}' % ', '.join(
                '%s. %s' % (author.first_names[0], author.surname)
                for author in author_list
                )
        author_cite_list = ['%s. %s' % (a.first_names[0], a.surname) 
                                for a in author_list]
        if len(author_cite_list) > 4:
            author_cite_list = author_cite_list[:3]
            author_cite_list.append('et al.')
        citation = ', '.join(author_cite_list) + \
        'in Proc. SciPy 2010, S. J. van der Walt, J. Millman, G. Varoquaux (Eds) '
        copyright = '\\copyright 2010, %s' % ( ', '.join(author_cite_list))
    else:
        authors = ''
        citation = 'Citation'
        copyright = 'Copyright'
    if hasattr(abstract, 'num_pages'):
        citation += 'pp. %i--%i' % (start_page, start_page +
                                        abstract.num_pages)
    else:
        citation += 'p. %i'% start_page
    if hasattr(abstract, 'number'):
        abstract.url = 'http://conference.scipy.org/proceedings/%s/paper_%i' \
        % (conf_name, abstract.number)
        url = r'\url{%s}' % abstract.url
    else:
        url = ''
    paper_text = abstract.paper_text
    if paper_text == '':
        paper_text = abstract.summary
    # XXX: It doesn't seem to be right to be doing this, but I get a
    # nasty UnicodeDecodeError on some rare abstracts, elsewhere.
    paper_text = codecs.utf_8_decode(hack_include_graphics(
                                rst2latex(paper_text)))[0]
    paper_abstract = abstract.paper_abstract
    if paper_abstract is None:
        paper_abstract = ''
    if not paper_abstract=='':
        paper_abstract = ur'\begin{abstract}%s\end{abstract}' % \
                    paper_abstract#.encode('utf-8')
    abstract_dict = {
            'text': paper_text.encode('utf-8'),
            'abstract': paper_abstract.encode('utf-8'),
            'authors': authors.encode('utf-8'),
            'title': abstract.title.encode('utf-8'),
            'citation': citation.encode('utf-8'),
            'copyright': copyright.encode('utf-8'),
            'url': url.encode('utf-8'),
        }
    outfile.write(codecs.utf_8_decode(ur'''
\phantomsection
\hypertarget{chapter}{} 
\vspace*{-2em}

\resetheadings{%(title)s}{%(citation)s}{%(url)s}{%(copyright)s}
\title{%(title)s}

\begin{minipage}{\linewidth}
%(authors)s
\end{minipage}

\noindent\rule{\linewidth}{0.2ex}
\vspace*{-0.5ex}
\twocolumngrid
%(abstract)s

\sloppy

%(text)s

\fussy
\onecolumngrid
\smallskip
\vfill
\filbreak
\clearpage

'''.encode('utf-8') % abstract_dict )[0])

def copy_files(dest=outfilename):
    """ Copy the required file from the source dir to the output dir.
    """
    dirname = os.path.dirname(dest)
    if dirname == '':
        dirname = '.'
    for filename in glob(sourcedir+os.sep+'*'):
        destfile = os.path.abspath(dirname + os.sep +
                                os.path.basename(filename))
        shutil.copy2(filename, destfile)
                            


def mk_abstract_preview(abstract, outfilename, attach_dir, start_page=None):
    """ Generate a preview for an given paper.
    """
    copy_files()
    for f in glob(os.path.join(attach_dir, '*')):
        if os.path.isdir(f):
            continue
        else:
            if not outdir == os.path.dirname(os.path.abspath(f)):
                shutil.copy2(f, outdir)
    for f in glob(os.path.join(sourcedir, '*')):
        if  os.path.isdir(f):
            os.makedirs(f)
        else:
            destfile = os.path.abspath(os.path.join(outdir, f))
            shutil.copy2(f, outdir)

    outbasename = outdir + os.sep + os.path.splitext(outfilename)[0]
    outfilename = outbasename + '.tex'

    outfile = codecs.open(outfilename, 'w', 'utf-8')
    preamble(outfile)
    render_abstract(outfile, abstract, start_page=start_page)
    outfile.write(ur'\end{document}' + u'\n')
    outfile.close()

    tex2pdf(outbasename, remove_tex=False)
#    abstract.num_pages = count_pages(outbasename + '.pdf')

    # Generate the tex file again, now that we know the length.
    outfile = codecs.open(outfilename, 'w', 'utf-8')
    preamble(outfile)
    render_abstract(outfile, abstract, start_page=start_page)
    outfile.write(ur'\end{document}' + u'\n')
    outfile.close()

    return tex2pdf(os.path.splitext(outfilename)[0], remove_tex=False)


##############################################################################
# Code for using outside of the webapp.
##############################################################################

def mk_zipfile():
    """ Generates a zipfile with the required files to build an
        abstract.
    """
    from zipfile import ZipFile
    zipfilename = os.path.join(os.path.dirname(__file__), 
                            'mk_scipy_paper.zip')
    z = ZipFile(zipfilename, 'w')
    for filename in glob(os.path.join(sourcedir, '*')):
        if not os.path.isdir(filename):
            z.write(filename, arcname='source/' + os.path.basename(filename))
    z.write(__file__, arcname='mk_scipy_paper.py')
    return zipfilename

class Bunch(dict):
    def __init__(self, **kwargs):
        dict.__init__(self, **kwargs)
        self.__dict__ = self

    def __reprt(self):
        return repr(self.__dict__)

author_like = Bunch(
        first_names='XX', 
        surname='XXX',
        email_address='xxx@XXX',
        institution='XXX',
        address='XXX',
        country='XXX'
)


abstract_like = Bunch(
        paper_abstract='An abstract',
        authors=[author_like, ],
        title='',
    )

if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-o", "--output", dest="outfilename",
                    default="./paper.pdf",
                    help="output to FILE", metavar="FILE")
    parser.usage = """%prog [options] rst_file [data_file]
    Compiles a given rest file and information file to pdf for the SciPy
    proceedings.
    """
    
    (options, args) = parser.parse_args()
    if not len(args) in (1, 2):
        print "One or two arguments required: the input rest file and " \
                "the input data file"
        print ''
        parser.print_help()
        sys.exit(1)
    infile = args[0]
    if len(args)==1:
        data_file = 'data.py'
        if os.path.exists('data.py'):
            print "Using data file 'data.py'"
        else:
            print "Generating the data file and storing it in data.py"
            print "You will need to edit this file to add title, author " \
                "information, and abstract."
            abstract = abstract_like
            file('data.py', 'w').write(repr(abstract))
    elif len(args)==2:
        data_file = args[1]
    
    abstract = Bunch( **eval(file(data_file).read()))
    abstract.authors = [Bunch(**a) for a in abstract.authors]

    abstract['summary'] = u''
    abstract['paper_text'] = file(infile).read().decode('utf-8')

    outfilename = options.outfilename

    mk_abstract_preview(abstract, options.outfilename, 
                            os.path.dirname(options.outfilename))
    # Ugly, but I don't want to wait on the thread.
    sys.exit()
