from __future__ import unicode_literals

__all__ = ['writer']

import docutils.core as dc
import docutils.writers
from docutils import nodes

from docutils.writers.latex2e import (Writer, LaTeXTranslator,
                                      PreambleCmds)

from .rstmath import mathEnv
from . import code_block

from options import options

try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

PreambleCmds.float_settings = u'''
\\usepackage[font={small,it},labelfont=bf]{caption}
\\usepackage{float}
'''

class Translator(LaTeXTranslator):
    def __init__(self, *args, **kwargs):
        LaTeXTranslator.__init__(self, *args, **kwargs)

        # Handle author declarations

        self.current_field = u''

        self.copyright_holder = None
        self.author_names = []
        self.author_institutions = []
        self.author_institution_map = dict()
        self.author_emails = []
        self.corresponding = []
        self.equal_contributors = []
        self.paper_title = u''
        self.abstract_text = []
        self.keywords = u''
        self.table_caption = []
        self.video_url = u''
        self.bibliography = u''

        # This gets read by the underlying docutils implementation.
        # If present, it is a list with the first entry the style name
        # and the second entry the BiBTeX file (see `visit_field_body`)
        self.bibtex = None

        self.abstract_in_progress = False
        self.non_breaking_paragraph = False

        self.figure_type = u'figure'
        self.figure_alignment = u'left'
        self.table_type = u'table'

        self.active_table.set_table_style(u'booktabs')

    def visit_docinfo(self, node):
        pass

    def depart_docinfo(self, node):
        pass

    def visit_author(self, node):
        self.author_names.append(self.encode(node.astext()))
        self.author_institution_map[self.author_names[-1]] = []
        raise nodes.SkipNode

    def depart_author(self, node):
        pass

    def visit_classifier(self, node):
        pass

    def depart_classifier(self, node):
        pass

    def visit_field_name(self, node):
        self.current_field = node.astext()
        raise nodes.SkipNode

    def visit_field_body(self, node):
        try:
            text = self.encode(node.astext())
        except TypeError:
            text = u''

        if self.current_field == u'email':
            self.author_emails.append(text)
        elif self.current_field == u'corresponding':
            self.corresponding.append(self.author_names[-1])
        elif self.current_field == u'equal-contributor':
            self.equal_contributors.append(self.author_names[-1])
        elif self.current_field == u'institution':
            self.author_institutions.append(text)
            self.author_institution_map[self.author_names[-1]].append(text)
        elif self.current_field == u'copyright_holder':
            self.copyright_holder = text
        elif self.current_field == u'video':
            self.video_url = text
        elif self.current_field == u'bibliography':
            self.bibtex = [u'alphaurl', text]
            self._use_latex_citations = True
            self._bibitems = [u'', u'']
            self.bibliography = text

        self.current_field = u''

        raise nodes.SkipNode

    def depart_field_body(self, node):
        raise nodes.SkipNode

    def depart_document(self, node):
        LaTeXTranslator.depart_document(self, node)

        ## Generate footmarks

        # build map: institution -> (author1, author2)
        institution_authors = OrderedDict()
        for auth in self.author_institution_map:
            for inst in self.author_institution_map[auth]:
                institution_authors.setdefault(inst, []).append(auth)

        def footmark(n):
            u"""Insert footmark #n.  Footmark 1 is reserved for
            the corresponding author. Footmark 2 is reserved for
            the equal contributors.\
            """
            return (u'\\setcounter{footnotecounter}{%d}' % n,
                    u'\\fnsymbol{footnotecounter}')

        # Build a footmark for the corresponding author
        corresponding_footmark = footmark(1)

        # Build a footmark for equal contributors
        equal_footmark = footmark(2)

        # Build one footmark for each institution
        institute_footmark = {}
        for i, inst in enumerate(institution_authors):
            institute_footmark[inst] = footmark(i + 3)

        footmark_template = ur'\thanks{%(footmark)s %(instutions)}'
        corresponding_auth_template = ur'''%%
          %(footmark_counter)s\thanks{%(footmark)s %%
          Corresponding author: \protect\href{mailto:%(email)s}{%(email)s}}'''

        equal_contrib_template = ur'''%%
          %(footmark_counter)s\thanks{%(footmark)s %%
          These authors contributed equally.}'''

        title = self.paper_title
        authors = []
        institutions_mentioned = set()
        equal_authors_mentioned = False
        corr_emails = []
        if len(self.corresponding) == 0:
            self.corresponding = [self.author_names[0]]
        for n, auth in enumerate(self.author_names):
            if auth in self.corresponding:
                corr_emails.append(self.author_emails[n])

        for n, auth in enumerate(self.author_names):
            # get footmarks
            footmarks = u''.join([u''.join(institute_footmark[inst]) for inst in self.author_institution_map[auth]])
            if auth in self.equal_contributors:
                footmarks += u''.join(equal_footmark)
            if auth in self.corresponding:
                footmarks += u''.join(corresponding_footmark)
            authors += [ur'%(author)s$^{%(footmark)s}$' %
                        {u'author': auth,
                        u'footmark': footmarks}]

            if auth in self.equal_contributors and equal_authors_mentioned==False:
                fm_counter, fm = equal_footmark
                authors[-1] += equal_contrib_template % \
                    {u'footmark_counter': fm_counter,
                     u'footmark': fm}
                equal_authors_mentioned = True

            if auth in self.corresponding:
                fm_counter, fm = corresponding_footmark
                authors[-1] += corresponding_auth_template % \
                    {u'footmark_counter': fm_counter,
                     u'footmark': fm,
                     u'email': ', '.join(corr_emails)}

            for inst in self.author_institution_map[auth]:
                if not inst in institutions_mentioned:
                    fm_counter, fm = institute_footmark[inst]
                    authors[-1] += ur'%(footmark_counter)s\thanks{%(footmark)s %(institution)s}' % \
                                {u'footmark_counter': fm_counter,
                                 u'footmark': fm,
                                 u'institution': inst}

                institutions_mentioned.add(inst)

        ## Add copyright

        # If things went spectacularly wrong, we could not even parse author
        # info.  Just fill in some dummy info so that we can see the error
        # messages in the resulting PDF.
        if len(self.author_names) == 0:
            self.author_names = [u'John Doe']
            self.author_emails = [u'john@doe.com']
            authors = [u'']

        copyright_holder = self.copyright_holder or (self.author_names[0] + (u'.' if len(self.author_names) == 1 else u' et al.'))
        author_notes = ur'''%%

          \noindent%%
          Copyright\,\copyright\,%(year)s %(copyright_holder)s %(copyright)s%%
        ''' % \
        {u'email': self.author_emails[0],
         u'year': options[u'proceedings'][u'year'],
         u'copyright_holder': copyright_holder,
         u'copyright': options[u'proceedings'][u'copyright'][u'article']}

        authors[-1] += ur'\thanks{%s}' % author_notes


        ## Set up title and page headers

        if not self.video_url:
            video_template = u''
        else:
            video_template = u'\\\\\\vspace{5mm}\\tt\\url{%s}\\vspace{-5mm}' % self.video_url

        title_template = ur'\newcounter{footnotecounter}' \
                ur'\title{%s}\author{%s' \
                ur'%s}\maketitle'
        title_template = title_template % (title, u', '.join(authors),
                                           video_template)

        marks = ur'''
          \renewcommand{\leftmark}{%s}
          \renewcommand{\rightmark}{%s}
        ''' % (options[u'proceedings'][u'title'][u'short'], title.upper())
        title_template += marks

        self.body_pre_docinfo = [title_template]

        # Save paper stats
        self.document.stats = {u'title': title,
                               u'authors': ', '.join(self.author_names),
                               u'author': self.author_names,
                               u'author_email': self.author_emails,
                               u'author_institution': self.author_institutions,
                               u'author_institution_map' : self.author_institution_map,
                               u'abstract': self.abstract_text,
                               u'keywords': self.keywords,
                               u'copyright_holder': copyright_holder,
                               u'video': self.video_url,
                               u'bibliography':self.bibliography}

        if hasattr(self, u'bibtex') and self.bibtex:
            self.document.stats.update({u'bibliography': self.bibtex[1]})

    def end_open_abstract(self, node):
        if u'abstract' not in node[u'classes'] and self.abstract_in_progress:
            self.out.append(u'\\end{abstract}')
            self.abstract_in_progress = False
        elif self.abstract_in_progress:
            self.abstract_text.append(self.encode(node.astext()))


    def visit_title(self, node):
        self.end_open_abstract(node)

        if self.section_level == 1:
            if self.paper_title:
                import warnings
                warnings.warn(RuntimeWarning(u"Title set twice--ignored. "
                                             u"Could be due to ReST"
                                             u"error.)"))
            else:
                self.paper_title = self.encode(node.astext())
            raise nodes.SkipNode

        elif node.astext() == u'References':
            raise nodes.SkipNode

        LaTeXTranslator.visit_title(self, node)

    def visit_paragraph(self, node):
        self.end_open_abstract(node)

        if u'abstract' in node[u'classes'] and not self.abstract_in_progress:
            self.out.append(u'\\begin{abstract}')
            self.abstract_text.append(self.encode(node.astext()))
            self.abstract_in_progress = True

        elif u'keywords' in node[u'classes']:
            self.out.append(u'\\begin{IEEEkeywords}')
            self.keywords = self.encode(node.astext())

        elif self.non_breaking_paragraph:
            self.non_breaking_paragraph = False

        else:
            if self.active_table.is_open():
                self.out.append(u'\n')
            else:
                self.out.append(u'\n\n')

    def depart_paragraph(self, node):
        if u'keywords' in node[u'classes']:
            self.out.append(u'\\end{IEEEkeywords}')

    def visit_figure(self, node):
        self.requirements[u'float_settings'] = PreambleCmds.float_settings

        self.figure_type = u'figure'
        if u'classes' in node.attributes:
            placements = u'[%s]' % u''.join(node.attributes[u'classes'])
            if u'w' in placements:
                placements = placements.replace(u'w', u'')
                self.figure_type = u'figure*'

        self.out.append(u'\\begin{%s}%s' % (self.figure_type, placements))

        if node.get(u'ids'):
            self.out += ['\n'] + self.ids_to_labels(node)

        self.figure_alignment = node.attributes.get(u'align', u'center')

    def depart_figure(self, node):
        self.out.append(u'\\end{%s}' % self.figure_type)

    def visit_image(self, node):
        align = self.figure_alignment or u'center'
        scale = node.attributes.get(u'scale', None)
        filename = node.attributes[u'uri']

        if self.figure_type == u'figure*':
            width = ur'\textwidth'
        else:
            width = ur'\columnwidth'

        figure_opts = []

        if scale is not None:
            figure_opts.append(u'scale=%.2f' % (scale / 100.))

        # Only add \columnwidth if scale or width have not been specified.
        if u'scale' not in node.attributes and u'width' not in node.attributes:
            figure_opts.append(ur'width=\columnwidth')

        self.out.append(ur'\noindent\makebox[%s][%s]' % (width, align[0]))
        self.out.append(ur'{\includegraphics[%s]{%s}}' % (u','.join(figure_opts),
                                                         filename))

    def visit_footnote(self, node):
        # Handle case where footnote consists only of math
        if len(node.astext().split()) < 2:
            node.append(nodes.label(text=u'_abcdefghijklmno_'))

        # Work-around for a bug in docutils where
        # "%" is prepended to footnote text
        LaTeXTranslator.visit_footnote(self, node)
        self.out[-1] = self.out[1].strip(u'%')

        self.non_breaking_paragraph = True

    def visit_table(self, node):
        classes = node.attributes.get(u'classes', [])
        if u'w' in classes:
            self.table_type = u'table*'
        else:
            self.table_type = u'table'

        self.out.append(ur'\begin{%s}' % self.table_type)
        LaTeXTranslator.visit_table(self, node)

    def depart_table(self, node):
        LaTeXTranslator.depart_table(self, node)

        self.out.append(ur'\caption{%s}' % ''.join(self.table_caption))
        self.table_caption = []

        self.out.append(ur'\end{%s}' % self.table_type)
        self.active_table.set(u'preamble written', 1)
        self.active_table.set_table_style(u'booktabs')

    def visit_thead(self, node):
        # Store table caption locally and then remove it
        # from the table so that docutils doesn't render it
        # (in the wrong place)
        if self.active_table.caption:
            self.table_caption = self.active_table.caption
            self.active_table.caption = []

        opening = self.active_table.get_opening()
        opening = opening.replace(u'linewidth', u'tablewidth')
        self.active_table.get_opening = lambda: opening

        # For some reason, docutils want to process longtable headers twice.  I
        # don't trust this fix entirely, but it does the trick for now.
        self.active_table.need_recurse = lambda: False

        LaTeXTranslator.visit_thead(self, node)

    def depart_thead(self, node):
        LaTeXTranslator.depart_thead(self, node)

    def visit_literal_block(self, node):
        self.non_breaking_paragraph = True

        if u'language' in node.attributes:
            # do highlighting
            from pygments import highlight
            from pygments.lexers import PythonLexer, get_lexer_by_name
            from pygments.formatters import LatexFormatter

            extra_opts = u'fontsize=\\footnotesize'

            linenos = node.attributes.get(u'linenos', False)
            linenostart = node.attributes.get(u'linenostart', 1)
            if linenos:
                extra_opts += u',xleftmargin=2.25mm,numbersep=3pt'

            lexer = get_lexer_by_name(node.attributes[u'language'])
            tex = highlight(node.astext(), lexer,
                            LatexFormatter(linenos=linenos,
                                           linenostart=linenostart,
                                           verboptions=extra_opts))

            self.out.append(u"\\vspace{1mm}\n" + tex +
                            u"\\vspace{1mm}\n")
            raise nodes.SkipNode
        else:
            LaTeXTranslator.visit_literal_block(self, node)

    def depart_literal_block(self, node):
        LaTeXTranslator.depart_literal_block(self, node)


    def visit_block_quote(self, node):
        self.out.append(u'\\begin{quotation}')
        LaTeXTranslator.visit_block_quote(self, node)

    def depart_block_quote(self, node):
        LaTeXTranslator.depart_block_quote(self, node)
        self.out.append(u'\\end{quotation}')


    # Math directives from rstex

    def visit_InlineMath(self, node):
        self.requirements[u'amsmath'] = u'\\usepackage{amsmath}'
        self.out.append(u'$' + node[u'latex'] + u'$')
        raise nodes.SkipNode

    def visit_PartMath(self, node):
        self.requirements[u'amsmath'] = u'\\usepackage{amsmath}'
        self.out.append(mathEnv(node[u'latex'], node[u'label'], node[u'type']))
        self.non_breaking_paragraph = True
        raise nodes.SkipNode

    def visit_PartLaTeX(self, node):
        if node[u"usepackage"]:
            for package in node[u"usepackage"]:
                self.requirements[package] = u'\\usepackage{%s}' % package
        self.out.append(u"\n" + node[u'latex'] + u"\n")
        raise nodes.SkipNode


writer = Writer()
writer.translator_class = Translator
