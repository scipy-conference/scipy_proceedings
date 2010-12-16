__all__ = ['writer']

import docutils.core as dc
import docutils.writers

from docutils.writers.latex2e import Writer, LaTeXTranslator

class Translator(LaTeXTranslator):
    pass

writer = Writer()
writer.translator_class = Translator
