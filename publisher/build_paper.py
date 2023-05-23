#!/usr/bin/env python
from __future__ import print_function, unicode_literals

from copy import deepcopy
from collections import OrderedDict
import docutils.core as dc
import itertools
import os
import os.path
import sys
import re
import tempfile
import glob
import shutil
import subprocess
import io
import yaml

from distutils import dir_util

from build_template import _from_template
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


# explanations for some of this may be found here
# https://github.com/docutils/docutils/blob/32ab1dbede9f1a93bfde479ea508cf308e0c5e1d/docutils/docs/user/latex.txt
latex_header = r"""
\def\PY@reset{\let\PY@it=\relax \let\PY@bf=\relax%
    \let\PY@ul=\relax \let\PY@tc=\relax%
    \let\PY@bc=\relax \let\PY@ff=\relax}
\def\PY@tok#1{\csname PY@tok@#1\endcsname}
\def\PY@toks#1+{\ifx\relax#1\empty\else%
    \PY@tok{#1}\expandafter\PY@toks\fi}
\def\PY@do#1{\PY@bc{\PY@tc{\PY@ul{%
    \PY@it{\PY@bf{\PY@ff{#1}}}}}}}
\def\PY#1#2{\PY@reset\PY@toks#1+\relax+\PY@do{#2}}

\@namedef{PY@tok@w}{\def\PY@tc##1{\textcolor[rgb]{0.73,0.73,0.73}{##1}}}
\@namedef{PY@tok@c}{\let\PY@it=\textit\def\PY@tc##1{\textcolor[rgb]{0.25,0.50,0.56}{##1}}}
\@namedef{PY@tok@cp}{\def\PY@tc##1{\textcolor[rgb]{0.00,0.44,0.13}{##1}}}
\@namedef{PY@tok@cs}{\def\PY@tc##1{\textcolor[rgb]{0.25,0.50,0.56}{##1}}\def\PY@bc##1{{\setlength{\fboxsep}{0pt}\colorbox[rgb]{1.00,0.94,0.94}{\strut ##1}}}}
\@namedef{PY@tok@k}{\let\PY@bf=\textbf\def\PY@tc##1{\textcolor[rgb]{0.00,0.44,0.13}{##1}}}
\@namedef{PY@tok@kp}{\def\PY@tc##1{\textcolor[rgb]{0.00,0.44,0.13}{##1}}}
\@namedef{PY@tok@kt}{\def\PY@tc##1{\textcolor[rgb]{0.56,0.13,0.00}{##1}}}
\@namedef{PY@tok@o}{\def\PY@tc##1{\textcolor[rgb]{0.40,0.40,0.40}{##1}}}
\@namedef{PY@tok@ow}{\let\PY@bf=\textbf\def\PY@tc##1{\textcolor[rgb]{0.00,0.44,0.13}{##1}}}
\@namedef{PY@tok@nb}{\def\PY@tc##1{\textcolor[rgb]{0.00,0.44,0.13}{##1}}}
\@namedef{PY@tok@nf}{\def\PY@tc##1{\textcolor[rgb]{0.02,0.16,0.49}{##1}}}
\@namedef{PY@tok@nc}{\let\PY@bf=\textbf\def\PY@tc##1{\textcolor[rgb]{0.05,0.52,0.71}{##1}}}
\@namedef{PY@tok@nn}{\let\PY@bf=\textbf\def\PY@tc##1{\textcolor[rgb]{0.05,0.52,0.71}{##1}}}
\@namedef{PY@tok@ne}{\def\PY@tc##1{\textcolor[rgb]{0.00,0.44,0.13}{##1}}}
\@namedef{PY@tok@nv}{\def\PY@tc##1{\textcolor[rgb]{0.73,0.38,0.84}{##1}}}
\@namedef{PY@tok@no}{\def\PY@tc##1{\textcolor[rgb]{0.38,0.68,0.84}{##1}}}
\@namedef{PY@tok@nl}{\let\PY@bf=\textbf\def\PY@tc##1{\textcolor[rgb]{0.00,0.13,0.44}{##1}}}
\@namedef{PY@tok@ni}{\let\PY@bf=\textbf\def\PY@tc##1{\textcolor[rgb]{0.84,0.33,0.22}{##1}}}
\@namedef{PY@tok@na}{\def\PY@tc##1{\textcolor[rgb]{0.25,0.44,0.63}{##1}}}
\@namedef{PY@tok@nt}{\let\PY@bf=\textbf\def\PY@tc##1{\textcolor[rgb]{0.02,0.16,0.45}{##1}}}
\@namedef{PY@tok@nd}{\let\PY@bf=\textbf\def\PY@tc##1{\textcolor[rgb]{0.33,0.33,0.33}{##1}}}
\@namedef{PY@tok@s}{\def\PY@tc##1{\textcolor[rgb]{0.25,0.44,0.63}{##1}}}
\@namedef{PY@tok@sd}{\let\PY@it=\textit\def\PY@tc##1{\textcolor[rgb]{0.25,0.44,0.63}{##1}}}
\@namedef{PY@tok@si}{\let\PY@it=\textit\def\PY@tc##1{\textcolor[rgb]{0.44,0.63,0.82}{##1}}}
\@namedef{PY@tok@se}{\let\PY@bf=\textbf\def\PY@tc##1{\textcolor[rgb]{0.25,0.44,0.63}{##1}}}
\@namedef{PY@tok@sr}{\def\PY@tc##1{\textcolor[rgb]{0.14,0.33,0.53}{##1}}}
\@namedef{PY@tok@ss}{\def\PY@tc##1{\textcolor[rgb]{0.32,0.47,0.09}{##1}}}
\@namedef{PY@tok@sx}{\def\PY@tc##1{\textcolor[rgb]{0.78,0.36,0.04}{##1}}}
\@namedef{PY@tok@m}{\def\PY@tc##1{\textcolor[rgb]{0.13,0.50,0.31}{##1}}}
\@namedef{PY@tok@gh}{\let\PY@bf=\textbf\def\PY@tc##1{\textcolor[rgb]{0.00,0.00,0.50}{##1}}}
\@namedef{PY@tok@gu}{\let\PY@bf=\textbf\def\PY@tc##1{\textcolor[rgb]{0.50,0.00,0.50}{##1}}}
\@namedef{PY@tok@gd}{\def\PY@tc##1{\textcolor[rgb]{0.63,0.00,0.00}{##1}}}
\@namedef{PY@tok@gi}{\def\PY@tc##1{\textcolor[rgb]{0.00,0.63,0.00}{##1}}}
\@namedef{PY@tok@gr}{\def\PY@tc##1{\textcolor[rgb]{1.00,0.00,0.00}{##1}}}
\@namedef{PY@tok@ge}{\let\PY@it=\textit}
\@namedef{PY@tok@gs}{\let\PY@bf=\textbf}
\@namedef{PY@tok@gp}{\let\PY@bf=\textbf\def\PY@tc##1{\textcolor[rgb]{0.78,0.36,0.04}{##1}}}
\@namedef{PY@tok@go}{\def\PY@tc##1{\textcolor[rgb]{0.20,0.20,0.20}{##1}}}
\@namedef{PY@tok@gt}{\def\PY@tc##1{\textcolor[rgb]{0.00,0.27,0.87}{##1}}}
\@namedef{PY@tok@err}{\def\PY@bc##1{{\setlength{\fboxsep}{\string -\fboxrule}\fcolorbox[rgb]{1.00,0.00,0.00}{1,1,1}{\strut ##1}}}}
\@namedef{PY@tok@kc}{\let\PY@bf=\textbf\def\PY@tc##1{\textcolor[rgb]{0.00,0.44,0.13}{##1}}}
\@namedef{PY@tok@kd}{\let\PY@bf=\textbf\def\PY@tc##1{\textcolor[rgb]{0.00,0.44,0.13}{##1}}}
\@namedef{PY@tok@kn}{\let\PY@bf=\textbf\def\PY@tc##1{\textcolor[rgb]{0.00,0.44,0.13}{##1}}}
\@namedef{PY@tok@kr}{\let\PY@bf=\textbf\def\PY@tc##1{\textcolor[rgb]{0.00,0.44,0.13}{##1}}}
\@namedef{PY@tok@bp}{\def\PY@tc##1{\textcolor[rgb]{0.00,0.44,0.13}{##1}}}
\@namedef{PY@tok@fm}{\def\PY@tc##1{\textcolor[rgb]{0.02,0.16,0.49}{##1}}}
\@namedef{PY@tok@vc}{\def\PY@tc##1{\textcolor[rgb]{0.73,0.38,0.84}{##1}}}
\@namedef{PY@tok@vg}{\def\PY@tc##1{\textcolor[rgb]{0.73,0.38,0.84}{##1}}}
\@namedef{PY@tok@vi}{\def\PY@tc##1{\textcolor[rgb]{0.73,0.38,0.84}{##1}}}
\@namedef{PY@tok@vm}{\def\PY@tc##1{\textcolor[rgb]{0.73,0.38,0.84}{##1}}}
\@namedef{PY@tok@sa}{\def\PY@tc##1{\textcolor[rgb]{0.25,0.44,0.63}{##1}}}
\@namedef{PY@tok@sb}{\def\PY@tc##1{\textcolor[rgb]{0.25,0.44,0.63}{##1}}}
\@namedef{PY@tok@sc}{\def\PY@tc##1{\textcolor[rgb]{0.25,0.44,0.63}{##1}}}
\@namedef{PY@tok@dl}{\def\PY@tc##1{\textcolor[rgb]{0.25,0.44,0.63}{##1}}}
\@namedef{PY@tok@s2}{\def\PY@tc##1{\textcolor[rgb]{0.25,0.44,0.63}{##1}}}
\@namedef{PY@tok@sh}{\def\PY@tc##1{\textcolor[rgb]{0.25,0.44,0.63}{##1}}}
\@namedef{PY@tok@s1}{\def\PY@tc##1{\textcolor[rgb]{0.25,0.44,0.63}{##1}}}
\@namedef{PY@tok@mb}{\def\PY@tc##1{\textcolor[rgb]{0.13,0.50,0.31}{##1}}}
\@namedef{PY@tok@mf}{\def\PY@tc##1{\textcolor[rgb]{0.13,0.50,0.31}{##1}}}
\@namedef{PY@tok@mh}{\def\PY@tc##1{\textcolor[rgb]{0.13,0.50,0.31}{##1}}}
\@namedef{PY@tok@mi}{\def\PY@tc##1{\textcolor[rgb]{0.13,0.50,0.31}{##1}}}
\@namedef{PY@tok@il}{\def\PY@tc##1{\textcolor[rgb]{0.13,0.50,0.31}{##1}}}
\@namedef{PY@tok@mo}{\def\PY@tc##1{\textcolor[rgb]{0.13,0.50,0.31}{##1}}}
\@namedef{PY@tok@ch}{\let\PY@it=\textit\def\PY@tc##1{\textcolor[rgb]{0.25,0.50,0.56}{##1}}}
\@namedef{PY@tok@cm}{\let\PY@it=\textit\def\PY@tc##1{\textcolor[rgb]{0.25,0.50,0.56}{##1}}}
\@namedef{PY@tok@cpf}{\let\PY@it=\textit\def\PY@tc##1{\textcolor[rgb]{0.25,0.50,0.56}{##1}}}
\@namedef{PY@tok@c1}{\let\PY@it=\textit\def\PY@tc##1{\textcolor[rgb]{0.25,0.50,0.56}{##1}}}

\def\PYZbs{\char`\\}
\def\PYZus{\char`\_}
\def\PYZob{\char`\{}
\def\PYZcb{\char`\}}
\def\PYZca{\char`\^}
\def\PYZam{\char`\&}
\def\PYZlt{\char`\<}
\def\PYZgt{\char`\>}
\def\PYZsh{\char`\#}
\def\PYZpc{\char`\%}
\def\PYZdl{\char`\$}
\def\PYZhy{\char`\-}
\def\PYZsq{\char`\'}
\def\PYZdq{\char`\"}
\def\PYZti{\char`\~}
% for compatibility with earlier versions
\def\PYZat{@}
\def\PYZlb{[}
\def\PYZrb{]}
\makeatother


%%% User specified packages and stylesheets

%%% Fallback definitions for Docutils-specific commands
% basic code highlight:
\providecommand*\DUrolecomment[1]{\textcolor[rgb]{0.40,0.40,0.40}{#1}}
\providecommand*\DUroledeleted[1]{\textcolor[rgb]{0.40,0.40,0.40}{#1}}
\providecommand*\DUrolekeyword[1]{\textbf{#1}}
\providecommand*\DUrolestring[1]{\textit{#1}}
% numeric or symbol footnotes with hyperlinks
\providecommand*{\DUfootnotemark}[3]{%
  \raisebox{1em}{\hypertarget{#1}{}}%
  \hyperlink{#2}{\textsuperscript{#3}}%
}
\providecommand{\DUfootnotetext}[4]{%
  \begingroup%
  \renewcommand{\thefootnote}{%
    \protect\raisebox{1em}{\protect\hypertarget{#1}{}}%
    \protect\hyperlink{#2}{#3}}%
  \footnotetext{#4}%
  \endgroup%
}

% inline markup (custom roles)
% \DUrole{#1}{#2} tries \DUrole#1{#2}
\providecommand*{\DUrole}[2]{%
  % backwards compatibility: try \docutilsrole#1{#2}
  \ifcsname docutilsrole#1\endcsname%
    \csname docutilsrole#1\endcsname{#2}%
  \else
    \csname DUrole#1\endcsname{#2}%
  \fi%
}

% hyperlinks:
\ifthenelse{\isundefined{\hypersetup}}{
  \usepackage[colorlinks=true,linkcolor=blue,urlcolor=blue]{hyperref}
  \usepackage{bookmark}
  \urlstyle{same} % normal text font (alternatives: tt, rm, sf)
}{}


%%% Body
\begin{document}
\InputIfFileExists{page_numbers.tex}{}{}
\newcommand*{\docutilsroleref}{\ref}
\newcommand*{\docutilsrolelabel}{\label}
\newcommand*\DUrolecode[1]{#1}
\providecommand*\DUrolecite[1]{\cite{#1}}
"""

BUILD_SYSTEM_FILES = (
    "template.tex",
    "page_numbers.tex",
    "scipy.sty",
    "README.md",
    "status.sty",
)


def detect_paper_type(in_path: str) -> str:
    """Figure out if we have RsT or Tex
    """
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


def prepare_dir(in_path: str, out_path: str, start: int):
    """Copy required files to build dir
    """
    # clear out whatever cruft might be in the output dir
    shutil.rmtree(out_path)
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
    for filename in ['_static/scipy.sty']:
        scipy_style = os.path.join(base_dir, filename)
        shutil.copy(scipy_style, out_path)


def preprocess_metadata(meta: dict) -> dict:
    """Transform metadata to match docutils output
    """
    # this is a bit unwieldy, and could probably use a good refactor,
    # both here and in writer/__init__
    new = deepcopy(meta)

    # start by making the author names listing
    for a in new['authors']:
        i = a.get('institution')
        if not i:
            a['institution'] = []
        elif isinstance(i, str):
            a['institution'] = [i]
        elif isinstance(i, list):
            a['institution'] = i
    new['author'] = [a['name'] for a in new['authors']]

    # then listing for the emails and institutions
    new['author_email'] = [a['email'] for a in new['authors']]
    new['author_institution'] = list(itertools.chain(a['institution'] for a in new['authors']))

    # we need to keep track of the corresponding and equal-contributors
    # tags that might be in the metadata file
    new['corresponding'] = [a['name'] for a in new['authors'] if a.get('corresponding')]
    new['equal_contributors'] = [a['name'] for a in new['authors'] if a.get('equal-contributor')]

    # now create mappings so we can look up institutions and orcids by name
    new['author_institution_map'] = {a['name']: a['institution'].copy() for a in new['authors'] if a['institution']}
    new['author_orcid_map'] = {a['name']: a['orcid'] for a in new['authors'] if a.get('orcid')}
    institutions = {}
    order = 1
    for a in new['authors']:
        for i in a['institution']:
            if i not in institutions:
                institutions[i] = {'order': order, 'authors': []}
                order += 1
            institutions[i]['authors'].append(a)
    new['institutions'] = [{'name': k, 'order': v['order']} for k, v in institutions.items()]
    for a in new['authors']:
        numbers = [institutions[i]['order'] for i in a['institution']]
        if numbers:
            a['institution'] = numbers
        else:
            a.pop('institution')
    new['authors'] = ', '.join(new['author'])

    # if there is no specified copyright holder, make it the first author
    copyright_holder = new.get('copyright_holder')
    if copyright_holder is None:
        copyright_holder = new['author'][0] + ('.' if len(new['author']) == 1 else ' et al.')
    new['copyright_holder'] = copyright_holder

    # finally handle the keywords and abstract
    abstract = new.get('abstract', '')
    if isinstance(abstract, str):
        new['abstract'] = [abstract.strip()]
    elif isinstance(abstract, list):
        new['abstract'] = [s.strip() for s in abstract]
    keywords = new.get('keywords')
    if isinstance(keywords, str):
        new['keywords'] = keywords.strip()
    elif isinstance(keywords, list):
        new['keywords'] = ', '.join([s.strip() for s in keywords])

    return new


def prepare_metadata(out_path: str) -> dict:
    """Find the metadata file and preprocess it
    """
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
    # everything else uses this, so it's important that this happens
    # after we make everything on the latex side look like it came from
    # our docutils extensions
    stats_file = os.path.join(out_path, 'paper_stats.json')
    d = options.cfg2dict(stats_file)
    try:
        d.update(metadata)
        options.dict2cfg(d, stats_file)
    except AttributeError:
        print("Error: no paper configuration found")
    return metadata


def rst2tex(out_path):

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
        rst, = glob.glob(os.path.join(out_path, '*.rst'))
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


def _footmark(n: int) -> str:
    return ('\\setcounter{footnotecounter}{%d}' % n,
            '\\fnsymbol{footnotecounter}')


def _mangle_author_fields(metadata: dict, config: dict) -> str:
    """Create author / title / thanks blocks from metadata
    """
    # this is copied almost verbatim from writer/__init__

    # build map: institution -> (author1, author2)
    institution_authors = OrderedDict()
    for auth in metadata['author_institution_map']:
        for inst in metadata['author_institution_map'][auth]:
            institution_authors.setdefault(inst, []).append(auth)

    # Build a footmark for the corresponding author
    corresponding_footmark = _footmark(1)

    # Build a footmark for equal contributors
    equal_footmark = _footmark(2)

    # Build one footmark for each institution
    institute_footmark = {}
    for i, inst in enumerate(institution_authors):
        institute_footmark[inst] = _footmark(i + 3)

    footmark_template = r'\thanks{%(footmark)s %(instutions)}'
    corresponding_auth_template = r'''%%
        %(footmark_counter)s\thanks{%(footmark)s %%
        Corresponding author: \protect\href{mailto:%(email)s}{%(email)s}}'''

    equal_contrib_template = r'''%%
        %(footmark_counter)s\thanks{%(footmark)s %%
        These authors contributed equally.}'''

    title = metadata['title']
    authors = []
    institutions_mentioned = set()
    equal_authors_mentioned = False
    corr_emails = []

    if len(metadata['corresponding']) == 0:
        metadata['corresponding'] = [metadata['author'][0]]
    for n, auth in enumerate(metadata['author']):
        if auth in metadata['corresponding']:
            corr_emails.append(metadata['author_email'][n])

    for n, auth in enumerate(metadata['author']):
        # get footmarks
        footmarks = ''.join([''.join(institute_footmark[inst]) for inst in metadata['author_institution_map'].get(auth, [])])
        if auth in metadata['equal_contributors']:
            footmarks += ''.join(equal_footmark)
        if auth in metadata['corresponding']:
            footmarks += ''.join(corresponding_footmark)
        authors += [r'%(author)s$^{%(footmark)s}$' %
                    {'author': auth,
                    'footmark': footmarks}]

        if auth in metadata['equal_contributors'] and equal_authors_mentioned==False:
            fm_counter, fm = equal_footmark
            authors[-1] += equal_contrib_template % \
                {'footmark_counter': fm_counter,
                    'footmark': fm}
            equal_authors_mentioned = True

        if auth in metadata['corresponding']:
            fm_counter, fm = corresponding_footmark
            authors[-1] += corresponding_auth_template % \
                {'footmark_counter': fm_counter,
                    'footmark': fm,
                    'email': ', '.join(corr_emails)}

        for inst in metadata['author_institution_map'].get(auth, []):
            if not inst in institutions_mentioned:
                fm_counter, fm = institute_footmark[inst]
                authors[-1] += r'%(footmark_counter)s\thanks{%(footmark)s %(institution)s}' % \
                            {'footmark_counter': fm_counter,
                                'footmark': fm,
                                'institution': inst}

            institutions_mentioned.add(inst)

    copyright_holder = metadata['copyright_holder']

    author_notes = r'''%%

        \noindent%%
        Copyright\,\copyright\,%(year)s %(copyright_holder)s %(copyright)s%%
    ''' % \
    {'year': config['proceedings']['year'],
        'copyright_holder': copyright_holder,
        'copyright': config['proceedings']['copyright']['article']}

    authors[-1] += r'\thanks{%s}' % author_notes

    ## Set up title and page headers

    if metadata.get('video'):
        video_template = '\\\\\\vspace{5mm}\\tt\\url{%s}\\vspace{-5mm}' % metadata.get('video')
    else:
        video_template = ''

    title_template = r'\newcounter{footnotecounter}' \
            r'\title{%s}\author{%s' \
            r'%s}\maketitle'
    title_template = title_template % (title, ', '.join(authors),
                                        video_template)

    marks = r'''
        \renewcommand{\leftmark}{%s}
        \renewcommand{\rightmark}{%s}
    ''' % (config['proceedings']['title']['short'], title.upper())
    title_template += marks

    return title_template


def tex2tex(out_path: str, metadata: dict):
    # find the paper
    paper_paths = glob.glob(os.path.join(out_path, '*.tex'))
    paper_paths = [p for p in paper_paths if not (p.endswith(BUILD_SYSTEM_FILES))]
    if len(paper_paths) > 1:
        raise RuntimeError("Found more than one input .tex--not sure which "
                           "one to use.")
    else:
        paper_path = paper_paths[0]
    with io.open(paper_path, mode='r', encoding='utf-8') as f:
        content = f.read()
    metadata = deepcopy(metadata)
    metadata['body'] = content
    metadata['header'] = latex_header
    config = options.get_config()
    metadata['title'] = _mangle_author_fields(metadata, config)
    text = _from_template('paper.tex', metadata, template_type='raw')
    with io.open(os.path.join(out_path, 'paper.tex'), 'w') as f:
        f.write(text)


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


def build_paper(paper_id: str, start: int = 1):
    out_path = os.path.join(output_dir, paper_id)
    in_path = os.path.join(papers_dir, paper_id)
    print("Building:", paper_id)

    # do this early, so we fail before copying anything
    paper_type = detect_paper_type(in_path)

    # copy a bunch of stuff
    options.mkdir_p(out_path)
    prepare_dir(in_path, out_path, start)

    if paper_type == 'rst':
        rst2tex(out_path)
    elif paper_type == 'tex':
        metadata_path = prepare_metadata(out_path)
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
