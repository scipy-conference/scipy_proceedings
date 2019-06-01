:author: Geoffrey M. Poore
:email: gpoore@uu.edu
:institution: Union University
:bibliography: poore


=======================================
Codebraid: Live Code in Pandoc Markdown
=======================================


.. class:: abstract

   Codebraid is a Python program that executes inline code and code blocks in
   a Pandoc Markdown document, then inserts the output into the final
   document.  The final document can include the original code as well as the
   resulting stdout and stderr.  The stdout can be included verbatim or
   interpreted as Markdown, which makes it possible to generate parts of a
   document programmatically and enhances reproducibility.  If there are
   errors, Codebraid parses the stderr and includes it next to the appropriate
   code for easy debugging.  Supported languages include Python, Julia, R,
   Rust, and Bash.  Adding additional languages is as simple as creating a
   short configuration file.  A single document can involve multiple
   programming languages or multiple independent sessions per language.

.. class:: keywords

   literate programming, reproducibility, dynamic report generation


Introduction
============

Scientific and technical documents are increasingly written with software that
allows a mixture of text and executable code.  This approach to writing can
enhance reproducibility, simplify code documentation, and aid in automating
reports.  For those using Python, the Jupyter Notebook :cite:`Kluyver2016` is
a popular option.  Well over one hundred additional Jupyter kernels are
available, including Julia and R kernels.  For those using R, knitr
:cite:`Xie2015` is probably more familiar.  Recently, the reticulate
:cite:`reticulate` and JuliaCall :cite:`juliacall` packages for R have
significantly expanded knitr's Python and Julia capabilities.  Emacs users
have Org Babel :cite:`Schulte2011,Schulte2012`.  LaTeX users have my own
PythonTeX package :cite:`Poore2013,Poore2015`, among other options.

This paper introduces Codebraid, which can execute code in Pandoc Markdown
documents :cite:`pandoc` as part of the document build process.  Codebraid is
focused on maximizing the power of having live code in a document while
providing superior usability.  Although Codebraid can function as a standalone
system, it is not necessarily in competition with earlier software.  In the
future, it may be possible to integrate Codebraid with Jupyter kernels or with
knitr to share code execution capabilities, or use a graphical user interface
(GUI) such as a notebook so that Pandoc Markdown becomes only an underlying
file format.  In many ways, Codebraid ultimately represents an attempt to
reimagine what is possible in documents that combine text with live code, with
a focus on three primary areas.


More code
---------

Codebraid allows any number of programming languages to be used within a
single document.  For each language that is used, multiple independent
sessions are allowed, so that code can be separated into independent pieces
that are only re-executed when necessary.  This is similar to Org Babel's
capabilities.  Jupyter notebooks are limited to a single kernel per notebook,
and thus a single language.  (Cell magics do allow additional languages, but
only at the level of a single isolated cell run in a subprocess.)  Multiple
languages are only possible in a Jupyter notebook by using special kernels and
extensions (for example, :cite:`SoS,BeakerX`).  knitr is more flexible since
it allows simultaneous R, Python, and Julia, though it is still limited to a
single session for each of those languages.  (All other knitr language engines
are executed as single isolated code chunks, and thus are essentially
equivalent to Jupyter cell magics.)

.. https://bookdown.org/yihui/rmarkdown/language-engines.html

Codebraid can also execute both code blocks and inline code, with equivalent
capabilities in both cases.  Org Babel is similar, while knitr allows inline
execution but only for printing the output of expressions.  Jupyter notebooks
require an extension :cite:`JupyterPythonMarkdown` for inline code
capabilities similar to knitr's.

.. https://orgmode.org/worg/org-contrib/babel/intro.html

.. https://yihui.name/knitr/demo/output/


More display options
--------------------

For inline code or a code block, Codebraid can display any combination of the
original Markdown source, the code, stdout, and stderr—in any order.  Its
display options are consciously designed with documentation and tutorials in
mind.  Org Babel can display code and/or stdout, but stderr is not captured
for document inclusion.  Including stderr requires workarounds like
redirecting it to stdout.  Jupyter notebooks display code, stdout, stderr, and
rich output (when supported).  Hiding some or all of these when exporting is
not supported, though it is possible with extensions (for example,
:cite:`JupyterHideInput` plus a custom template) or additional software (for
example, :cite:`JupyterBook`).  knitr can display code and/or output, although
the order is fixed.

Another Codebraid feature is the ability to name inline code or a code block,
and then use that name elsewhere in the document to display any combination of
its original Markdown source, code, stdout, and stderr.  Org Babel can do this
with code and stdout, and knitr can do it with code and output.  Jupyter
notebooks lack such capabilities.

.. https://jupyter.org/jupyter-book/features/hiding.html

.. https://yihui.name/knitr/options/ -> ref.label


More (standard) plain text
--------------------------

Plain text document formats are convenient when writing documents with a
version control system, because they give simple diffs.  Codebraid uses Pandoc
Markdown, knitr works with Markdown as well as other formats like LaTeX, and
Org Babel relies on Org mode markup, so this is an advantage they share.
Jupyter notebooks are saved in JSON format, which has led to special diffing
tools such as :cite:`nbdime`.

Although plain text formats have advantages when it comes to diffing, they are
more complex to parse than a data format like JSON.  Because Org Babel is
integrated with Org mode markup, it requires no special parsing.  In contrast,
knitr extends the Markdown syntax for code and thus requires special parsing.
Similar programs like Pweave :cite:`pweave` and Weave.jl :cite:`weavejl` that
execute code in Markdown take the same approach.  This extended syntax is
handled by a preprocessor that extracts the code for execution and then
inserts the output into a copy of the original document, which can then be
processed with Pandoc or another parser.  While this approach minimizes the
overhead associated with code extraction and output insertion, the
preprocessor can introduce significant cognitive load for users.  For example,
knitr's preprocessor employs simple regex matching and does not understand
Markdown comments, so code in a commented-out part of a document still runs.
Writing tutorials that show literal knitr code chunks can involve inserting
empty strings, zero-width spaces, linebreaks, or Unicode escapes to avoid the
preprocessor's tendency to execute everything :cite:`knitrfaq,Hovorka`.

.. https://github.com/rstudio/rmarkdown/issues/974

.. https://github.com/yihui/knitr/issues/1363

.. https://rviews.rstudio.com/2017/12/04/how-to-show-r-inline-code-blocks-in-r-markdown/

.. https://yihui.name/knitr/faq/

Codebraid takes a different approach that guarantees correct results at the
expense of some additional parsing overhead.  Pandoc does not just convert
directly between documents in different formats.  It also provides an abstract
syntax tree (AST) as one output option.  Codebraid uses only standard Pandoc
Markdown syntax, so it is able to perform all code extraction and output
insertion as operations on the AST.  Since this approach requires running
Pandoc multiple times, it does involve more overhead than the preprocessor
approach.  However, since Pandoc is responsible for all Markdown parsing, even
edge cases for Markdown parsing can be handled correctly.

A simple example
================

A simple Pandoc Markdown document that runs code with Codebraid is shown
below.

.. code:: text

   ```{.python .cb.run name=part1}
   var1 = "Hello from *Python!*"
   var2 = f"Here is some math:  $2^8={2**8}$."
   ```

   ```{.python .cb.run name=part2}
   print(var1)
   print(var2)
   ```

..

Pandoc Markdown defines attributes for inline code and code blocks.
These have the general form

::

   {#id .class1 .class2 key1=value1 key2=value2}

If code with these attributes were converted into HTML, ``#id`` becomes
an HTML id for the code, anything with the form ``.class`` specifies
classes, and space-separated key-value pairs provide additional
attributes. Although key-value pairs can be quoted with double quotation
marks, Pandoc allows most characters except the space and equals sign
unquoted. Other output formats such as LaTeX use attributes in a largely
equivalent manner.

Pandoc uses the first class to determine the language name, hence the
``.python`` in the example above. Codebraid uses the second class to
specify a command for processing the code. All Codebraid commands are
under a ``cb`` namespace to prevent unintentional collisions with normal
Pandoc inline code and code blocks. In this case, ``cb.run`` indicates
that code should be run, stdout should be included and interpreted as
Markdown, and stderr should be displayed in the event of errors.
Finally, in this example, the ``name`` keyword is used to assign a
unique name to each piece of code. This allows the code to be referenced
elsewhere in a document to insert any combination of its Markdown
source, code, stdout, and stderr.

If this were a normal Pandoc document, converting it to a format such as
reStructuredText could be accomplished by running

::

   pandoc --from markdown --to rst file.md

Using Codebraid to execute code as part of the document conversion
process is as simple as replacing ``pandoc`` with ``codebraid pandoc``:

::

   codebraid pandoc --from markdown --to rst file.md

The ``codebraid`` executable is available from the Python Package Index
(PyPI); development is at https://github.com/gpoore/codebraid.

When this ``codebraid pandoc`` command is executed, the original
Markdown shown above is converted into the Codebraid-processed Markdown

.. code:: text

   Hello from *Python!*
   Here is some math:  $2^8=256$.

This processed Markdown would then be converted into the final
reStructuredText, rendering as

   Hello from *Python!* Here is some math: :math:`2^8=256`.

   ..

By default, the output of ``cb.run`` is interpreted as Markdown. It is
possible to show the output verbatim instead, as discussed later.

In this example, the code is simple enough that it could be executed
every time the document is built, but that will often not be the case.
By default, Codebraid caches all code output, and code is only
re-executed when it is modified. This can be changed by building with
the flag ``--no-cache``.

Creating examples
=================

The example in the last section was actually itself an example of using
Codebraid. This paper was written in Markdown, then converted to
reStructuredText via Codebraid with Pandoc. Finally, the
reStructuredText was converted through LaTeX to PDF via docutils. The
two code blocks in the example were only entered in the original
Markdown source of this paper a single time, and Codebraid only executed
them a single time. However, with Codebraid’s copy-paste capabilities,
it was possible to display the code and output in other locations in the
document programmatically.

The rendered output of the two code blocks is shown at the very end of
the last section. This is where the code blocks were actually entered in
the original Markdown source of this paper, and where they were
executed.

Recall that both blocks were given names, ``part1`` and ``part2``. This
enables any combination of their Markdown source, code, stdout, and
stderr to be inserted elsewhere in the document. At the beginning of the
previous section, the Markdown source for the blocks was shown. This was
accomplished via

.. code:: text

   ```{.cb.paste copy=part1+part2 show=copied_markup}
   ```

The ``cb.paste`` command inserts copied data from one or more code
chunks that are specified with the ``copy`` keyword. Meanwhile, the
``show`` keyword controls what is displayed. In this case, the Markdown
source of the copied code chunks was shown. Since the ``cb.paste``
command is copying content from elsewhere, it is used with an empty code
block. Alternatively, a single empty line or a single line containing an
underscore is allowed as a placeholder.

Toward the end of the last section, the verbatim output of the
Codebraid-processed Markdown was displayed. This was inserted in a
similar manner:

.. code:: text

   ```{.cb.paste copy=part1+part2 show=stdout:verbatim}
   ```

The default format of ``stdout`` is ``verbatim``, but this was specified
just to be explicit. The other option is ``raw``, or interpreted as
Markdown.

Of course, all Markdown shown in the current section was itself inserted
programmatically using ``cb.paste`` to copy from the previous section.
However, to prevent infinite recursion, the next section is not devoted
to explaining how this was accomplished.

Other Codebraid commands
========================

The commands ``cb.run`` and ``cb.paste`` have already been introduced.
There are three additional commands.

The ``cb.code`` command simply displays code, like normal inline code or
a code block. It primarily exists so that normal code can be named, and
then accessed later. ``cb.paste`` could be used to insert the code
elsewhere, perhaps combined with code from other sources via something
like ``copy=code1+code2``. It would also be possible to run the code
elsewhere:

::

   ```{.cb.run copy=code1+code2}
   ```

When ``copy`` is used with ``cb.run``, or another command that executes
code, only code is copied, and everything proceeds as if this code had
been entered directly in the code block.

The ``cb.expr`` command only works with inline code, unlike other
commands. It evaluates an expression and then prints a string
representation. For example,

.. code:: text

   `2**128`{.python .cb.expr}

produces

   340282366920938463463374607431768211456

As this demonstrates, Pandoc code attributes for inline code immediately
follow the closing backtick(s). While this sort of a “postfix” notation
may not be ideal from some from perspectives, it is the cost of
maintaining full compatibility with Pandoc Markdown syntax.

Finally, the ``cb.nb`` command runs code in “notebook mode.” For code
blocks, this displays code followed by verbatim stdout. If there are
errors, stderr is also included automatically. For inline code,
``cb.nb`` is equivalent to ``cb.expr``. The markdown

.. code:: text

   ```{.python .cb.nb name=notebook}
   import random
   random.seed(2)
   rnums = [random.randrange(100) for n in range(8)]
   print(f"Random numbers: {rnums}")
   print(f"Sorted numbers: {sorted(rnums)}")
   print(f"Range: {[min(rnums), max(rnums)]}")
   ```

results in

.. code:: python

   import random
   random.seed(2)
   rnums = [random.randrange(100) for n in range(8)]
   print(f"Random numbers: {rnums}")
   print(f"Sorted numbers: {sorted(rnums)}")
   print(f"Range: {[min(rnums), max(rnums)]}")

.. code:: text

   Random numbers: [7, 11, 10, 46, 21, 94, 85, 39]
   Sorted numbers: [7, 10, 11, 21, 39, 46, 85, 94]
   Range: [7, 94]

Display options
===============

There are two code chunk keywords that govern display, ``show`` and
``hide``. These can be used to override the default display settings for
each command.

``show`` takes any combination of the following options: ``markup``
(display Markdown source), ``code``, ``stdout``, ``stderr``, and
``none``. Multiple options can be combined, such as
``show=code+stdout+stderr``. Code chunks using ``copy`` can also employ
``copied_markup`` to display the Markdown source of the copied code.
When the ``cb.expr`` command is used, the expression output is available
via ``expr``. ``show`` completely overwrites the existing display
settings.

The display format can also be specified with ``show``. ``stdout``,
``stderr``, and ``expr`` can take the formats ``raw`` (interpreted as
Markdown), ``verbatim``, or ``verbatim_or_empty`` (verbatim if there is
output, otherwise a space or empty line). For example,
``show=stdout:raw+stderr:verbatim``. While a format can be specified for
``markup`` and ``code``, only the default ``verbatim`` is permitted.

``hide`` takes the same options as show, except that ``none`` is
replaced by ``all`` and formats are not specified. Instead of overriding
existing settings like ``show``, ``hide`` removes the specified options
from those that already exist.

Advanced code execution
=======================

Ideally, executable code should arranged within a document based on what
is best for the reader, rather than in a manner dictated by limitations
of the tooling. Several options are provided to maximize the flexibility
of code presentation.

Incomplete units of code
------------------------

By default, Codebraid requires that code be divided into complete units.
For example, a code block must contain an entire loop, or an entire
function definition. Codebraid can detect the presence of an incomplete
unit of code because it interferes with stdout and stderr processing, in
which case Codebraid will raise an error.

The ``complete`` keyword allows incomplete units of code. While this
increases the flexibility of code layout, it also means that any output
will not be shown until the next complete code chunk.

The Markdown for a somewhat contrived example that demonstrates these
capabilities is shown below, along with its output.

.. code:: text

   ```{.python .cb.run complete=false}
   for n in range(11):
   ```

   ```{.python .cb.run complete=false}
       if n % 2 == 0:
   ```

   ```{.python .cb.run}
           if n < 10:
               print(f"{n}, ", end="")
           else:
               print(f"{n}")
   ```

..

   0, 2, 4, 6, 8, 10

   ..

Sessions
--------

By default, all code for a language is executed within a single session,
so variables and data are shared between code chunks. It can be
convenient to separate code into multiple sessions when several
independent tasks are being performed, or when a long calculation is
required but the output can easily be saved and loaded by separate code
for visualization or other processing. The ``session`` keyword makes
this possible. For example,

.. code:: text

   ```{.python .cb.run session=long}
   import json
   result = sum(range(100_000_000))
   with open("result.json", "w") as f:
       json.dump({"result": result}, f)
   ```

All sessions are currently executed in serial. In the future, support
for parallel execution may be added.

Outside ``main()``
------------------

Codebraid executes code by inserting it into a template. The template
allows stdout and stderr to be broken into pieces and correctly
associated with the code chunks that created them. For a language like
Python under typical usage, ``complete`` eliminates the few limitations
of this approach. However, the situation for a compiled language with a
``main()`` function is more complex.

Codebraid includes support for Rust. By default, code is inserted into a
template that defines a ``main()`` function. Thus, a code block like

.. code:: text

   ```{.rust .cb.run}
   let x = "Greetings from *Rust!*";
   println!("{}", x);
   ```

can run to produce

   Greetings from *Rust!*

   ..

In some situations, it would be convenient to completely control the
definition of the ``main()`` function and add code outside of
``main()``. The ``outside_main`` keyword makes this possible. All code
chunks with ``outside_main=true`` at the beginning of a session are used
to overwrite the beginning of the ``main()`` template, while any chunks
with ``outside_main=true`` at the end of the session are used to
overwrite the end of the template. If all code chunks have
``outside_main=true``, then all of Codebraid’s templates are completely
omitted, and all output is associated with the final code chunk. The
example below demonstrates this option.

.. code:: text

   ```{.rust .cb.run outside_main=true}
   fn main() {
       use std::fmt::Write as FmtWrite;
       use std::io::Write as IoWrite;
       let x = "Rust says hello.  Again!";
       println!("{}", x);
   }
   ```

..

   Rust says hello. Again!

   ..

Working with external files
===========================

Though Codebraid is focused on embedding executable code within a
document, there will be times when it is useful to interact with
external source files. Since Codebraid processes code with a programming
language’s standard interpreter or compiler, normal module systems are
fully compatible; for example, in Python, ``import`` works normally.
Codebraid provides additional ways to work with external files via the
``include_file`` option.

When ``include_file`` is used with the ``cb.code`` command, an external
source file is simply included and displayed. It is possible to include
only certain line ranges using the additional option ``include_lines``,
or only part of a file that matches a regular expression via
``include_regex``. For example,

.. code:: text

   ```{.markdown .cb.code include_file=poore.txt
   include_regex="# Working.*?,"}
   ```

includes the original Markdown source for this paper, and then uses a
regular expression to display only the first few lines of the current
section:

.. code:: text

   # Working with external files

   Though Codebraid is focused on embedding executable
   code within a document,

Since the ``cb.code`` command is including content from elsewhere, it is
used with an empty code block. Alternatively, a single empty line or a
single line containing an underscore is allowed as a placeholder. This
example included part of a file using a single regular expression. There
are also options for including everything starting with or starting
after a literal string or regular expression, and for including
everything before or through a literal string or regular expression.

The ``include_file`` option works with commands that execute code as
well. For instance,

::

   ```{.python .cb.run include_file=code.py}
   ```

would read in the contents of an external file “code.py” and then run it
in the default Python session, just as if it had been entered directly
within the Markdown file.

Implementation and language support
===================================

Codebraid currently supports Python 3.5+, Julia, Rust, R, and Bash. This
section provides an overview of how code is executed and the procedure
for adding support for additional languages.

Unless ``outside_main=true`` or ``complete=false``, code is inserted
into a template before execution. The template writes delimiters to
stdout and stderr at the beginning of each code chunk. These delimiters
are based on a hash of the code to eliminate the potential for
collisions. Once execution is complete, Codebraid parses stdout and
stderr and uses these delimiters to associate output with individual
code chunks. This is why using ``outside_main=true`` or
``complete=false`` delays the inclusion of output to a later code chunk;
there are no delimiters. This system is a more advanced variant of the
one I created previously in PythonTeX :cite:`Poore2015`.

Each individual delimiter is unique, and is tracked individually by
Codebraid. This allows incomplete units of code that have not been
marked with ``complete=false`` to be detected. If this code interferes
with the template to produce an error, Codebraid can use the stderr
delimiters plus parsing of stderr to find the source. If the code does
not produce an error, but prevents a delimiter from being written or
causes a delimiter to be written multiple times or not at the beginning
of a line, this will also be detected and traced back. Under normal
conditions, interfering with the delimiters without detection requires
conscious effort.

Adding support for additional languages is simply a matter of creating
the necessary templates and putting them in a configuration file. Basic
language support can require very little, essentially just code for
writing the delimiters to stdout and stderr. For example, Bash support
is based on this three-line template:

::

   printf "\n{stdout_delim}\n"
   printf "\n{stderr_delim}\n" >&2
   {code}

The Bash configuration file also specifies that the file extension
``.sh`` should be used, and provides another four lines of template code
to enable ``cb.expr``. So far, the longest configuration file, for Rust,
is less than fifty lines—counting empty lines.

Debugging
=========

Because code is typically inserted into a template for execution, if
there are errors the line numbers will not correspond to those of the
code that was extracted from the document, but rather to those of the
code that was actually executed. Codebraid tracks line numbers during
template assembly, so that executed line numbers can be converted into
original line numbers. Then it parses stderr and corrects line numbers.
An example of an error produced with ``cb.nb`` with Python is shown
below.

.. code:: python

   var = 123
   print(var, flush=True)
   var += "a"

.. code:: text

   123

.. code:: text

   Traceback (most recent call last):
     File "source.py", line 3, in <module>
       var += "a"
   TypeError: unsupported operand type(s) for +=:
   'int' and 'str'

..

Since line numbers in errors and warnings correspond to those in the
code entered by the user, and since anything written to stderr is
displayed by default next to the code that caused it, debugging is
significantly simplified. In many cases, this even applies to compile
errors, as can be demonstrated with some Rust code in a new session.
First, define a variable:

.. code:: rust

   let number = 123;

Then introduce a syntax error:

.. code:: rust

   number -/

.. code:: text

   error: expected expression, found `/`
     --> source.rs:2:9
      |
    2 | number -/
      |         ^ expected expression

   error: aborting due to previous error

The compile error appears next to the code that caused it, with a line
number of 2, which is appropriate since this is the second line of code
in this session.

Another source of errors is invalid code chunk options or an invalid
combination of options. In these cases, Codebraid omits everything that
would normally be displayed and instead provides an error message. This
includes the line number in the Markdown source where the error
occurred. The Pandoc AST does not currently contain source information.
Instead, Codebraid performs a parallel string search through the
Markdown source and the AST to associate code with line numbers in the
Markdown source.


Conclusion
==========

Codebraid provides a unique and powerful combination of features for executing
code embedded in Pandoc Markdown documents.  A single document can contain
multiple languages and multiple independent sessions per language.  Any
combination of Markdown source, code, stdout, and stderr can be displayed, and
it is easy to reuse code and output elsewhere in a document.

There are several logical avenues for further development.  One of the
original motivations for creating Codebraid was to build on my previous work
with PythonTeX :cite:`Poore2015` to create a code execution system that could
be used with multiple markup languages.  While Codebraid has focused thus far
on Pandoc Markdown, little of it is actually Markdown-specific.  It should be
possible to work with other markup languages supported by Pandoc, so long as
Pandoc parses key-value attributes for some variant of a code block in those
languages.  Pandoc has recently added Jupyter notebooks to its extensive list
of supported formats.  Perhaps at some point it may even be conceivable to
convert a Codebraid document into a Jupyter notebook, perform some exploratory
programming for a single session of a single language, and then convert back
to Markdown.

Another, simpler integration with Jupyter would be to add support for running
code using Jupyter kernels rather than Codebraid's built-in system.
Codebraid's multiple independent sessions give it advantages for some types of
computations, but there are times when the responsiveness of a Jupyter kernel
would be convenient.  Pweave :cite:`pweave` has previously used Jupyter
kernels to execute code extracted from Markdown documents.

Codebraid's caching system could also be improved in the future.  Currently,
caching is based only on the code that is executed.  Adding a way to specify
external dependencies such as data files would be beneficial.

.. noweb and literate programming?
