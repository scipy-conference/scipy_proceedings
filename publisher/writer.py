__all__ = ['writer']

import docutils.core as dc
import docutils.writers
from docutils import nodes

from docutils.writers.latex2e import (Writer, LaTeXTranslator,
                                      PreambleCmds)

class Translator(LaTeXTranslator):
    def __init__(self, *args, **kwargs):
        LaTeXTranslator.__init__(self, *args, **kwargs)

    # Handle author declarations

    current_field = ''

    author_names = []
    author_institutions = []
    author_emails = []
    paper_title = ''

    def visit_docinfo(self, node):
        pass

    def depart_docinfo(self, node):
        pass

    def visit_author(self, node):
        self.author_names.append(self.encode(node.astext()))
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
        text = self.encode(node.astext())

        if self.current_field == 'email':
            self.author_emails.append(text)
        elif self.current_field == 'institution':
            self.author_institutions.append(text)

        self.current_field = ''

        raise nodes.SkipNode

    def depart_field_body(self, node):
        raise nodes.SkipNode

    def depart_document(self, node):
        LaTeXTranslator.depart_document(self, node)

        doc_title = '\\title{%s}' % self.paper_title
        doc_title += '\\author{%s}' % ', '.join(self.author_names)
        doc_title += '\\maketitle'

        self.body_pre_docinfo = [doc_title]

    def visit_title(self, node):
        if self.section_level == 1:
            self.paper_title = self.encode(node.astext())
            raise nodes.SkipNode

        LaTeXTranslator.visit_title(self, node)

    def visit_paragraph(self, node):
        if 'abstract' in node['classes']:
            self.out.append('\\begin{abstract}')

    def depart_paragraph(self, node):
        if 'abstract' in node['classes']:
            self.out.append('\\end{abstract}')

writer = Writer()
writer.translator_class = Translator
