# This code is from: http://pypi.python.org/pypi/rstex/

#!/usr/bin/python2
from docutils import utils, nodes
from docutils.core import publish_cmdline
from docutils.writers.latex2e import Writer, LaTeXTranslator
from docutils.parsers.rst import roles, Directive, directives


class InlineMath(nodes.Inline, nodes.TextElement):
    pass

class InlineRef(nodes.Inline, nodes.TextElement):
    pass

class InlineCite(nodes.Inline, nodes.TextElement):
    pass

class PartMath(nodes.Part, nodes.Element):
    pass

class PartLaTeX(nodes.Part, nodes.Element):
    pass

class PartBibliography(nodes.Part, nodes.Element):
    pass



class RsTeXTranslator(LaTeXTranslator):

    def visit_InlineMath(self, node):
        self.requirements['amsmath'] = r'\usepackage{amsmath}'
        self.body.append('$' + node['latex'] + '$')
        raise nodes.SkipNode

    def visit_InlineRef(self, node):
        self.body.append('\\eqref{%s}' % node['target'])
        raise nodes.SkipNode

    def visit_InlineCite(self, node):
        #self.requirements['natbib'] = r'\usepackage{natbib}'
        self.body.append('\\cite{%s}' % node['latex'])
        raise nodes.SkipNode

    def visit_PartMath(self, node):
        self.requirements['amsmath'] = r'\usepackage{amsmath}'
        self.body.append(mathEnv(node['latex'], node['label'], node['type']))
        raise nodes.SkipNode

    def visit_PartLaTeX(self, node):
        if node["usepackage"]:
            for package in node["usepackage"]:
                self.requirements[package] = r'\usepackage{%s}' % package
        self.body.append("\n" + node['latex'] + "\n")
        raise nodes.SkipNode

    def visit_PartBibliography(self, node):
        self.body.append("\n" +
                         "\\bibliographystyle{%s}\n" % node["style"] +
                         "\\bibliography{%s}\n" % node["references"] +
                         "\n")

        raise nodes.SkipNode


class RsTeXWriter(Writer):

    def __init__(self):
        Writer.__init__(self)
        self.translator_class = RsTeXTranslator

def mathEnv(math, label, type):
    if type in ("split", "gathered"):
        begin = "\\begin{equation}\n\\begin{%s}\n" % type
        end = "\\end{%s}\n\\end{equation}\n" % type
    else:
        begin = "\\begin{%s}\n" % type
        end = "\\end{%s}\n" % type
    if label:
        begin += "\\label{%s}\n" % label
    return begin + math + "\n" + end

def mathRole(role, rawtext, text, lineno, inliner, options={}, content=[]):
    latex = utils.unescape(text, restore_backslashes=True)
    return [InlineMath(latex=latex)], []

def refRole(role, rawtext, text, lineno, inliner, options={}, content=[]):
    text = utils.unescape(text)
    node = InlineRef('(?)', '(?)', target=text)
    return [node], []

def citeRole(role, rawtext, text, lineno, inliner, options={}, content=[]):
    latex = utils.unescape(text, restore_backslashes=True)
    return [InlineCite(latex=latex)], []


class MathDirective(Directive):

    has_content = True
    required_arguments = 0
    optional_arguments = 2
    final_argument_whitespace = True
    option_spec = {
        'type': directives.unchanged,
        'label': directives.unchanged,
    }
    def run(self):
        latex = '\n'.join(self.content)
        if self.arguments and self.arguments[0]:
            latex = self.arguments[0] + '\n\n' + latex
        node = PartMath()
        node['latex'] = latex
        node['label'] = self.options.get('label', None)
        node['type'] = self.options.get('type', "equation")
        ret = [node]
        return ret

class LaTeXDirective(Directive):

    has_content = True
    required_arguments = 0
    optional_arguments = 1
    final_argument_whitespace = True
    option_spec = {
        'usepackage': directives.unchanged
    }
    def run(self):
        latex = '\n'.join(self.content)
        if self.arguments and self.arguments[0]:
            latex = self.arguments[0] + '\n\n' + latex
        node = PartLaTeX()
        node['latex'] = latex
        node['usepackage'] = self.options.get("usepackage", "").split(",")
        ret = [node]
        return ret


class BibliographyDirective(Directive):

    has_content = False
    required_arguments = 1 # bib source
    optional_arguments = 1 # style
    final_argument_whitespace = True
    option_spec = {
        'style': directives.unchanged
    }


    def run(self):
        references = directives.uri(self.arguments[0])
        node = PartBibliography()
        node['references'] = references
        node['style'] = self.options.get("style", "plain")
        ret = [node]
        return ret





roles.register_local_role("math", mathRole)
roles.register_local_role("ref", refRole)
roles.register_local_role("cite", citeRole)
directives.register_directive("math", MathDirective)
directives.register_directive("latex", LaTeXDirective)
directives.register_directive("bibliography", BibliographyDirective)
publish_cmdline(writer=RsTeXWriter())
