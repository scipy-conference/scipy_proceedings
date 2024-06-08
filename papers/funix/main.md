---
# Ensure that this title is the same as the one in `myst.yml`
title: Funix - vivifying (any) Python functions into GUI apps
abstract: |
  The raise of machine learning (ML) and artificial intelligence (AI), especially the generative AI (GenAI), has brought up the need for wrapping models or algorithms into GUI apps. For example, a large language model (LLM) can be access through a string-to-string GUI app with a textbox as the primary input.
  Most of existing solutions require developers to manually create widgets and link them to arguments/returns of a function individually. This low-level process is laborious and usually intrusive. Funix takes a CSS-like approach to this problem by automatically picking widgets based on the types of the arguments and returns of a function according to the type-to-widget mapping defined in a theme. For example, in Funix' default theme, the Python native types `str`, `bool`, and `Literal` are mapped into an input box, a checkbox, and a set of radio buttons in the MUI library. As a result, an existing Python function can be turned into a GUI app without any code change. Unlike existing solutions that bound the choice of widgets to those provided by them, as a transcompiler, Funix allows a developer to pick any component in the frontend world and customize it in JSON without any knowledge to JavaScript/TypeScript. Funix further uses the information in docstrings, which are common in Python development, to control the appearance of the GUI. 
---

## Introduction

Presenting a model or an algorithm as a GUI app is a common need in the scientific and engineering community.
For example, a large language model (LLM) is not accessible to the general public until we wrap it with a chat interface, which consists of a text input and a text output.
Because most scientists and engineers are not familiar with frontend development which is JavaScript/TypeScript-centric, there have been many solutions based on Python, one of the most popular programming languages in scientific computing -- especially AI, such as [ipywidgets](https://ipywidgets.readthedocs.io/en/stable/), [Streamlit](https://streamlit.io/), [Gradio](https://www.gradio.app/), [Reflex](https://reflex.dev/), [Dash](https://dash.plotly.com/), and [PyWebIO](https://www.pyweb.io/).
Most of them follow the conventional GUI programming philosophy that a developer needs to manually pick widgets from a widget library and associate them with the arguments and returns of an underlying function, which is usually called the "call back function."

This approach has several drawbacks. **First**, it needs continuous manual work to align the GUI code with the callback function. If the signature of the callback function changes, the GUI code needs to be manually updated. In this sense, the developer is manually maintaining two sets of code that are highly overlapping. **Second**, the GUI configuration is low-level, at individual widgets. It is difficult to reuse the design choices and thus laborious to keep the appearance of the GUI consistent across apps. **Third**, the developer is bounded by the widgets provided by the GUI library. If there is an unsupported datatype, the developer most likely has to give up due to the lack of frontend development knowledge. **Fourth**, they do not leverage the features of the Python language itself to reduce the amount of additional work. For example, it is very common in Python development to use docstrings to annotate arguments of a function. However, most of the existing solutions still requires the developer to manually specify the labels of the widgets. **Last** but not least, all existing solutions require the developer to read their documentations before being able to say "hello, world!"

[Funix](http://funix.io), which stands for "function/funny" + "*nix", takes a different approach. We notice that the choice of a widget has a weak correlation with the type of the function I/O it is associated with. For example, a checkbox is unsuitable for any type other than Boolean. Therefore, Funix takes a CSS-like approach that automatically picks the widgets based on the types of the arguments/returns of a function. The type-to-widget mapping is defined in a theme. For example, in the default theme of Funix, the Python native types `str`, `bool`, and `Literal`, respectively, are mapped into an input box, a checkbox, and a set of radio buttons in the MUI library, while the common scientific types `pandas.DataFrame` and `matplotlib.figure.Figure` are mapped to tables and charts.

```{code} python
:label: code_hello_world
:caption: A basic hello, world! example in Funix. It is an ordinary Python function with nothing special to Funix. The corresponding GUI app is shown in [](#fig_hello_world).

# hello.py
def hello(your_name: str) -> str:
    return f"Hello, {your_name}."
```

```{figure} hello.png
:label: fig_hello_world
:align: center

The app generated from [](#code_hello_world) by the command `funix hello.py` by Funix.
```

```{code} python
:label: code_advanced_input_widgets
:caption: An advanced input widgets example in Funix. The corresponding GUI app is shown in [](#fig_advanced_input_widgets). Note that 

# hello.py
import typing # Python native 

import ipywidgets  # popular UI library 

def input_widgets_basic(
    prompt: str = "Who is Oppenheimer?",
    advanced_features: bool = True,
    model: typing.Literal['GPT-3.5', 'GPT-4.0', 'Falcon-7B'] = 'GPT-4.0',
    max_token: range(100, 200, 20) = 140,
    openai_key: ipywidgets.Password = "1234556",
    )  -> str:
    pass
```

```{figure} advanced_input_widgets.png
:label: fig_advanced_input_widgets
:align: center

The input panel of the app generated from [](#code_advanced_input_widgets) by Funix. 
```

As a result, an existing Python function can be turned into a GUI app without any code change. The type-to-widget mapping makes it easy to keep the GUI appearance consistent across apps. Scientific developers can focus on building the functions that best use their domain knowledge and leave the rest to Funix, and if any, the customized themes designed by their UI teams.
Although Funix uses themes to control the UI, customizations that cannot be done via types, e.g., rate limiting, can be done via a Funix decorator.

Funix also exploits other features or common practices in Python programming to make building apps more intuitive and efficient. default values, Docstring and multiple pages app.

Funix's architecture: a transcompiler that exposes the frontend world to Python developers.
Funix is more than a GUI generator. As a transcompiler, it produces both the frontend ...The backend can be access programmatically. It is based on websocket...It's not only the React code or the UI. It is an actual app that can run.
The two panels...Y

Funix can be really useful in building what we call "disposable apps" -- apps that are built for a short-live purposes, such as collecting human evaluation of an AI model, collecting human labeling of AI training data, exhibit the capabilities of an API, etc.

In summary, Funix has the following distinctive features compared with many existing solutions:

1. Type-based GUI generation controlled by themes
2. Exposing the frontend world to Python developers
3. Leveraging Python's language features to make building apps more intuitively. In other words, less new stuff from Funix as possible.

The rest of the paper is organized as follows:
In [Motivation](#motivation-a-balance-between-simplicity-and-versatility), we discuss the motivation behind Funix and what are the cases suitable and not suitable for Funix. ...

## Motivation: balancing simplicity and versatility for GUI app development

When it comes to GUI app development, there is a trade-off between simplicity and versatility.
JavaScript/TypeScript-based web frontend frameworks like React, Angular, and Vue.js are versatile.
But that versatility is beyond the reach of most scientists and engineers, except frontend/full-stack engineers, and is usually overkilling for most scientific and engineering apps.

As machine learning researchers ourselves, we notice that a great (if not overwhemling) amount of scientific apps have the following two features:

1. the underlying logic is a straightforward input-output process -- thus complex interactivity, such as updating the input options based on existing user input, is not needed;
2. the app is not the goal but a necessary step to the goal -- thus it is not worthy to spend time on building the app.

Most existing solutions do a great job for the first feature above but are still too complicated for the second one, requiring a developer to read their documentations and add some code before an app can be fired up.
Since versatility is already given up for simplicity in Python-based app building, why not trade it further for more simplicity?

Funix pushes the simplicity to the extreme.
In the "hello, world" example ([](#code_hello_world)) above, you get an app without learning anything nor modifying the code.
To pursue the simplicity goal, Funix squeezes information from what is already common in Python development, such as typing hints (or type inference) and docstrings save developers from extra work or learning. Funix does not need or want to become a Domain Specific Language (DSL). The Python programming language itself (including typing hint syntax and docstring styles) is already a surface language for GUI app building.

We would also like to argue that the raise of GenAI is simplifying the GUI as natural languages are becoming a prominent interface between humans and computers. Consider text-to-image generation, the app only needs a string input and an image output. In this sense, Funix, and its Python-based peers, will be able to meet a lot of needs, in scientific computing or general, in the future.

Because Funix is designed for quickly firing up apps that model a straightforward input-output process, it has one input panel on the left and an output panel on the right. The underlying function will be called after the user plugs in the arguments and click the "Run" button. The result will be displayed in the output panel.

## Funix's default mapping from Python types to React components

Like CSS, Funix controls the GUI appearance based on the types of arguments and returns of a function.
Under the hood, Funix transcompiles the signature of a function into React code.
Currently, Funix depends on the type hint for every argument or return of a function. In the future, it will support type inference or tracing.

By default, Funix maps the following basic Python types to the following MUI components:

| Python type     | Input or Output | MUI component or HTML                                                                                                                                                                 |
|-----------------|-----------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `str`           | Input           | [TextField](https://mui.com/material-ui/react-text-field/)                                                                                                                            |
| `bool`          | Input           | [Checkbox](https://mui.com/material-ui/react-checkbox/) or [Switch](https://mui.com/material-ui/react-switch/)                                                                        |
| `int`           | Input           | [TextField](https://mui.com/material-ui/react-text-field/)                                                                                                                            |
| `float`         | Input           | [TextField](https://mui.com/material-ui/react-text-field/) or [Slider](https://mui.com/material-ui/react-slider/)                                                                     |
| `Literal`       | Input           | [RadioGroup](https://mui.com/material-ui/react-radio-button/) if number of elements is below 8; [Select](https://mui.com/material-ui/react-select/) otherwise                         |
| `range`         | Input           | [Slider](https://mui.com/material-ui/react-slider/)                                                                                                                                   |
| `List[Literal]` | Input           | An array of [Checkboxes](https://mui.com/material-ui/react-checkbox/) if the number of elements is below 8; [AutoComplete](https://mui.com/material-ui/react-autocomplete/) otherwise |
| `str`           | Output          | Plain text                                                                                                                                                                            |
| `bool`          | Output          | Plain text                                                                                                                                                                            |
| `int`           | Output          | Plain text                                                                                                                                                                            |
| `float`         | Output          | Plain text                                                                                                                                                                            |

In particular, we leverage the semantics of `Literal` and `List[Literal]` for single-choice and multiple-choice selections.

Because Funix is a transcompiler, it benefits from multimedia types defined in popular Python libraries. For example, the following `ipywidgets` and `IPython` types are natively supported by Funix -- although they are mapped to MUI components rather than `ipywidgets` (Jupyter's input widgets) or `IPython` (Jupyter's display system) components:

| Python type                                     | Input or Output | MUI component or HTML                                                             |
|-------------------------------------------------|-----------------|-----------------------------------------------------------------------------------|
| `ipywidgets.Password`                           | Input           | [TextField](https://mui.com/material-ui/react-text-field/) with `type="password"` |
| `ipywidgets.Image`                              | Input           | [React Dropzone](https://react-dropzone.js.org/) combine with MUI Components      |
| `ipywidgets.Video`                              | Input           | [React Dropzone](https://react-dropzone.js.org/) combine with MUI Components      |
| `ipywidgets.Audio`                              | Input           | [React Dropzone](https://react-dropzone.js.org/) combine with MUI Components      |
| `ipywidgets.FileUpload`                         | Input           | [React Dropzone](https://react-dropzone.js.org/) combine with MUI Components      |
| `IPython.display.HTML`                          | Output          | Raw HTML                                                                          |
| `IPython.display.Markdown`                      | Output          | [React Markdown](https://github.com/remarkjs/react-markdown)                      |
| `IPython.display.JavaScript`                    | Output          | Raw JavaScript                                                                    |
| `IPython.display.Image`                         | Output          | [CardMedia](https://mui.com/material-ui/react-card/#media) with `component=img`   |
| `IPython.display.Video`                         | Output          | [CardMedia](https://mui.com/material-ui/react-card/#media) with `component=video` |
| `IPython.display.Audio`                         | Output          | [CardMedia](https://mui.com/material-ui/react-card/#media) with `component=audio` |
| `matplotlib.figure.Figure`                      | Output          | [mpld3](https://mpld3.github.io/)                                                 |
| `pandas.DataFrame` & `pandera.typing.DataFrame` | Input & Output  | [DataGrid](https://mui.com/x/react-data-grid/)                                    |

Funix also has built-in support to `pandas.DataFrame` and `matplotlib.figure.Figure` as we mentioned above, which are mapped to tables and charts. For example, the function below

```python
import pandas, matplotlib.pyplot
from numpy import arange, log
from numpy.random import random

def table_and_plot(
   df: pandas.DataFrame = pandas.DataFrame({
       "a": arange(500) + random(500)/5,
       "b": random(500)-0.5 + log(arange(500)+1),
       "c": log(arange(500)+1) })
   ) -> matplotlib.figure.Figure:

   fig = matplotlib.pyplot.figure()
   matplotlib.pyplot.plot(df["a"], df["b"], 'b')
   matplotlib.pyplot.plot(df["a"], df["c"], 'r')

   return fig
```

becomes the app below:

![Table and Chart](table_and_chart.png)

where both the table and the chart are interactive/editable. As far as we know, no other Python-based solutions supports editable tables as inputs.

## Defining new type-to-widget mappings

:::{note}
Introducing a new type-to-widget mapping or modifying an existing one should not be the job of most Funix users. Only advanced users or the UI guys should be involved. This is like most scientists who writes papers in LaTeX do not develop the LaTeX classes or macros but just use them.
:::

<!-- Funix provides two ways, [the decorator way](#the-decorator-way) and [the theme way](#the-theme+way) to bind a type, whether it is Python-native, from a third party, or user-defined, to a widget.  -->
<!-- In addition to type-based, automatic widget selection, Funix also allows manual, per-variable widget selection as in its peer solutions. Please refer to the [decorator](#decorator) section for more details. -->

As a transcompiler, Funix does not have its own widget library. Instead, it exposes theoretically any React component library to Funix users by giving them full freedom to bind a type to a React component on the street. As to be seen below, it further allows configuring the properties of widgets.
All of these can be done without any knowledge of JavaScript/TypeScript or React.
In this sense, Funix bridges the Python world with the frontend world and made Python or JSON the surface language for (React-based) frontend development.

### Using the `new_funix_type` decorator

A new type can be introduced as simple as a new class. The mapping from the class to a widget can be then defined via the `widget` parameter of the `new_funix_type` decorator of `funix`. [](#code_new_type) defines a new type `blackout`, which a special case (as indicated by the inheritance) of `str` and binds it with a widget.

Funix follows the convention in the frontend world to identify a widget by a module specifier in `npm`, the de facto package manager in the frontend world. In [](#code_new_type), the widget is identified as `@mui/material/TextField`. Properties of the widget supported by its library for configuration can be passed into the `new_funix_type` decorator. As mentioned earlier, this allows a Pythonista to tap into a React component library without frontend development knowledge.

```{code} python
:label: code_new_type
:caption: An example of introducing a new type, binding it to a widget, and using it. 

from funix import new_funix_type
@new_funix_type(
  widget = {
      "widget": "@mui/material/TextField",
      "props": {
          "type": "password",
          "placeholder": "Enter a secret here."
      }
  }
)
class blackout(str):
   def print(self):
       return self + " is the message."

def hoho(x: blackout = "Funix Rocks!") -> str:
   return x.print()
```

### Using the theme

A type-to-widget mapping can be reused and centralized managed via a theme, which is a simple JSON file. An example is given in [](#code_theme) below where the Python's native types `str`, `int`, and `float` are bound to three widgets. In this exmaple, besides using `npm` module specifier, Funix shorthand strings `inputbox` and `slider` are also used.

```{code} jsonc
:label: code_theme
:caption: An example theme. 
{
 "name": "grandma's secret theme", // space and punctuation allowed
 "widgets": {    
   "str": "inputbox",        // Funix' shorthand, non-parametric
   "int": "slider[0,100,2]", // Funix' shorthand, parametric
   "float": {
       "widget": "@mui/material/Slider", 
       // using MUI's widget
       // https://mui.com/material-ui/api/slider
       "props": { 
           // config props of the frontend widget
           "min": 0,
           "max": 100,
           "step": 0.1
       }
   }
 }
}
```

## Building apps Pythonically

Funix leverages the language features of Python and common practices in Python development to make app building in Python more intuitive and efficient.

### Default values as placeholders

Python supports default values for keyword arguments. Funix directly uses them as the placeholder values for corresponding widgets. In contrast, Funix' peer solutions require developers to provide the placeholder values the second time in the widget initiation.

### Making use of Docstrings

Docstrings are dominantly common in the Python community.
Information in Docstrings is often needed in the GUI of an app.
For example, the annotation of each argument can be displayed as a label/tooltip to explain the meaning of the argument to the app user.
Therefore, Funix automatically adds selected information from Docstrings into apps, instead of requiring the developer to do it again as in peer solutions.

Different sections of Docstrings will become different kinds of information on the frontend.
The Docstring above the first headed section (`Args` in [](#code_docstring)) will be rendered as Markdown. Argument annotations in the `Args` section will become the labels in the UI. Finally, the `Examples` section will become prefilled example values for a user to try out.
Funix currently only supports Google-style and Numpy-style docstrings.

Only information in the `Args` and `Examples` section will be displayed in the GUI.

```{code} python
:label: code_docstring
:caption: An example of a function with a Google-style docstring. The corresponding GUI app is shown in [](#fig_docstring).

def foo(x: str, y: int) -> str:
    """## What happens when you multiply a string with an integer?    

    Try it out below. 

    Parameters
    ----------
    x : str
        A string that you want to repeat. 
    y : int
        How many times you want to repeat. 
    
    Examples
    ----------
    >>> foo("hello ", 3)
    "hello hello hello "
    """
    return x * y
```

```{figure} docstring.png
:label: fig_docstring
:align: center

An app with input panel customized by Docstrings. 
```

### Output layout in `print` and `return`

In Funix, by default, the return of a function becomes the content of the output panel.
A user can control the layout of the output panel returning strings, including and especially f-strings, in the Markdown and HTML syntaxes.
Markdown and HTML strings must be explicitly specified of the types `IPython.display.Markdown` and `IPython.display.HTML`, respectively. Otherwise, the raw strings will be displayed. Because Python supports multiple returns, you can mix Markdown and HTML strings in the return statement.

Quite often we need to print out some information before a function reaches its returns.
`print` is a Python built-in function that is frequently used for this purpose.
Funix extends this convenience by redirecting the output of `print` to the output panel of an app. The printout strings in Markdown or HTML syntax and will be automatically rendered after syntax detection.
To avoid conflicting with the default behavior of printing to `stdout`, printing to the web needs to be explicitly turned on by a decorator.

```{code} python
:label: code_print_return
:caption: Use `print` and `return` to control the output layout. The corresponding GUI app is shown in [](#fig_print_return).

from IPython.display import Markdown, HTML
from typing import Tuple
from funix import funix

@funix(print_to_web=True)
def foo(income: int = 200000, tax_rate: float= 0.45) -> Tuple[Markdown, HTML]:
    print (f"## Here is your tax breakdown: \n")

    tax_table = (
        " | Item | $$$ | \n"
        " | --- | --- |  \n"
        f"| Income | {income} | \n"
        f"| Tax | -{tax_rate * income : .3f} | \n"
        f"| Net Income | {income - tax_rate * income : .3f} | \n\n"
    )
  
    return tax_table, "If you have any question, contact <a href='http://irs.gov'>IRS</a>."
```

```{figure} print_return.png
```

### Streaming based on `yield`

In this GenAI era, streaming is becoming a common way for returning lengthy text output from an AI model.
Instead of inventing something new to support streaming, Funix repurposes the `yield` keyword in Python to stream the output of a function. The rationale is that `return` and `yield` are highly similar and `return` has already been used to display the final output in a Funix-powered app.

```{code} python
:label: code_stream
:caption: Stremaing using yield in Funix. The corresponding GUI app is shown in [](#fig_stream).
import time

def stream() -> str:
    """
    ## Streaming demo in Funix

    To see it, simply click the "Run" button.
    """
    message = "We the People of the United States, in Order to form a more perfect Union, establish Justice, insure domestic Tranquility, provide for the common defence, promote the general Welfare, and secure the Blessings of Liberty to ourselves and our Posterity, do ordain and establish this Constitution for the United States of America. "

    for i in range(len(message)):
        time.sleep(0.01)
        yield message[0:i]
```

```{figure} stream.gif
:label: fig_stream
:align: center

The streaming demo in Funix. Source code in [](#code_stream). 
```

### States, sessions, and multi-page apps

In Funix, maintaining states is as simple as updating a `global` variable.
By leveraging the semantics of `global` variables which are a native feature of the Python language, Funix saves developers the burden of learning something new in Funix.

```{code} python
:label: code_hangman
:caption: A simple Hangman game in Funix that uses the `global` keyword to maintain the state. This solution is much shorter than using peer solutions, such as in [Gradio](https://www.gradio.app/guides/state-in-blocks). 

from IPython.display import Markdown

secret_word = "funix"
used_letters = [] # a global variable to maintain the state/session

def guess_letter(Enter_a_letter: str) -> Markdown:
    letter = Enter_a_letter # rename
    global used_letters # state/session as global
    used_letters.append(letter)
    answer = "".join([
        (letter if letter in used_letters else "_")
            for letter in secret_word
        ])
    return f"### Hangman \n `{answer}` \n\n ---- \n ### Used letters \n {', '.join(used_letters)}"
```

A security risk is that there is only one backend server for one Funix app and consequently a `global` variable is accessible by all browser sessions of the app. Sessionization can eliminate this risk. Simple start up the Funix app with a `-t` flag.

A special but useful case that needs to maintain states is multi-page apps. A multi-page app is an app of many pages/tabs that share the same variable space. Values of variables set in one page can be accessed in another page.
Without reinventing the wheel, Funix supports this need by turning a Python class into a multi-page app, where each member function becomes a page of the multi-page app and pages can exchange data via the `self` variable. In particular, the constructor (`__init__`) becomes the landing page of the multi-page app. Because each instance of the class is independent, the multi-page app is sessionized for different connections from browsers.
In this approach, a Funix developer does not have to learn anything new but can start easily from the OOP principles that they are already familiar with.

```{code} python
:label: code_class
:caption: A simple multi-page app in Funix leveraging OOP. The corresponding GUI app is shown in [](#fig_stream).

from funix import funix_method, funix_class
from IPython.display import Markdown, HTML

@funix_class()
class A:
    @funix_method(print_to_web=True)
    def __init__(self, a: int):
        self.a = a
        print(f"`self.a` has been initialized to {self.a}")

    @funix_method()
    def set(self, b: int) -> Markdown:
        """Update the value for `self.a`. """
        old_a  = self.a
        self.a = b
        return (
                "| var | value |\n" 
                "| ----| ------|\n"
            f"| `a` before | {old_a} |\n" 
            f"| `a` after | {self.a} |"
            )

    @funix_method()
    def get(self) -> HTML:
        """Check the value of `self.a`. """
        return f"The value of <code>self.a</code> is <i>{self.a}</i>. "
```

```{figure} class.gif
:label: fig_class

A multiplage app generated by Funix from a class of three member methods including the constructor. Source code in [](#code_class)
```

## The Funix decorators

Although Funix relies on the Python language (including type hints and docstrings) itself to define GUI apps,
there are still some aspects on the appearance and behavior of an apps uncovered. There is where the `@funix` decorator kicks. One example above is redirecting the `print` from `stdout` to the output panel of an app. Here we just show a few more examples.

Funix uses types to determine the widgets. However, there may be occasions that modifying the typing-to-widget mapping may  not be worth it, for cases like exceptions. The `@funix` dectorator has a `widgets` parameter for this purpose. The `widgets` parameter takes the same value as in a Funix theme. [](#code_sentence_builder) is an example to temporarily override the widget choice for `List[Literal]`.

As mentioned earlier, Funix is suitable for straightforward input-output processes. Such a process is triggered once when the "Run" button is clicked. This may work for many cases but in many other cases, we may want the output to be updated following the changes in the input end automatically.
To do so, simply toggle on the `autorun` parameter in the `@funix` decorator. This will activate the "continuously run" checkbox on the input panel.

```{code}
:label: code_autorun
:caption: Auto run. 

import matplotlib.pyplot, matplotlib.figure
import numpy
from funix import funix


@funix(autorun=True, widgets={"omega": "slider[0, 4, 0.1]"})
def sine(omega: float = 0) -> matplotlib.figure.Figure:
    fig = matplotlib.pyplot.figure()
    x = numpy.linspace(0, 20, 200)
    y = numpy.sin(x * omega)
    matplotlib.pyplot.plot(x, y, linewidth=5)
    return fig

```

```{figure} autorun.gif
:label: fig_autorun
```

Although interactivity is not a strong suit of Funix for reasons aforementioned, Funix still provides some support to simple but frequently interactivity needs. It can reveal some widgets only when certain conditions are met. This is called "conditional visibility" in Funix.

```{code}
:label: code_conditional_visible
:caption: Conditional visibility in `@funix` decorator 

import typing
import openai
import funix

openai.api_key = os.environ.get("OPENAI_KEY")

@funix.funix(
  conditional_visible=[
    {
      "when": {"show_advanced": True,},
      "show": ["max_tokens", "model", "openai_key"]
    }
  ]
)
def ChatGPT_advanced(
  prompt: str,
  show_advanced: bool = False,
  model : typing.Literal['gpt-3.5-turbo', 'gpt-3.5-turbo-0301']= 'gpt-3.5-turbo',
  max_tokens: range(100, 200, 20)=140,
  openai_key: str = ""
) -> str:
  completion = openai.ChatCompletion.create(
    messages=[{"role": "user", "content": prompt}],
    model=model,
    max_tokens=max_tokens,
  )
  return completion["choices"][0]["message"]["content"]
```

```{figure} conditional_visible.gif

```

When an app is exposed, a common concern is how to avoid abuses. Rate limiting is a common measure to this. Funix's `@funix` decorator supports rate limiting based on both browser sessions and time.

Funix does not convert private functions by default, so if some normally-named functions don't need to be converted (e.g., util functions), you can turn off exporting to the website with the `disable` parameter.

Funix can dynamically calculate other arguments based on what the user has not yet submitted, but is entering. This is extremely helpful for tax rate calculations, etc.

```{code}
:label: code_reactive
:caption: Reactive

from funix import funix

@funix(disable=True)
def compute_tax(salary: float, income_tax_rate: float) -> int:
    return salary * income_tax_rate


@funix(
  reactive={"tax": compute_tax}
)
def after_tax_income_calculator(
    salary: float, 
    income_tax_rate: float, 
    tax: float) -> str:
    return f"Your take home money is {salary - tax} dollars,\
    for a salary of {salary} dollars, \
    after a {income_tax_rate*100}% income tax."
```

```{figure} reactive.gif
:label: fig_reactive
```

Lastly, togging on `show_source` parameter in `@funix` can enable the source code of your app to be displayed.

## Jupyter support

Jupyter is a popular tool for Python development. Funix supports turning a Python function/class defined in a Jupyter cell into an app inside Jupyter.

```{code}
:label: code_jupyter
:caption: Jupyter

# cell 1
from funix import funix

# cell 2
@funix()
def hello(name: str) -> str:
    return f"Hello, {name}!"
```

```{figure} jupyter.png
:label: fig_jupyter
```

## Showcases

Lastly, please allow us to use some examples to demonstrate the convenient and power of Funix in quickly prototyping apps. If there is any frontend knowledge needed, it is only HTML. 

1. Wordle. 

   The source code can be found [here](https://github.com/TexteaInc/funix/blob/develop/examples/games/wordle.py). In Funix, only simple HTML code that changes the background colors for tiles of letters according to the rule of the game Wordle is needed. A GIF showing the game in action is in [](#fig_wordle). 
  
    ```{figure} wordle.gif
    :label: fig_wordle

    The Wordle game implemented in Funix. Source code [here](https://github.com/TexteaInc/funix/blob/develop/examples/games/wordle.py).
    ```
  
2. Dall-E. 

   This example shows how easy to wrap a GenAI API into a GUI app. Once again, there is nothing special to Funix. Just plain Python code. 
  
    ```{code} python
    :label: code_dalle
    :caption: Source code for the Dall-E app in Funix. The corresponding GUI app is shown in [](#fig_dalle).
    
    import openai  # pip install openai
    import IPython 

    def dalle(Prompt: str = "a flying cat on a jet plane") 
              -> IPython.display.Image:

    client = openai.OpenAI() # defaults to os.environ.get("OPENAI_API_KEY")

    response = client.images.generate(
        prompt=Prompt,
    )

    return response.data[0].url
    ```
  
    ```{figure} dalle.png
    :label: fig_dalle
    ```
  
3. ChatGPT multi-turn

   Funix does not have a chat widget, because it is so easy to build one by yourself using simple alignment controls in HTML. In this way, a developer has full control rather than being bounded by the widgets provided by a GUI library. [](#code_joke) is a simple example of a multi-turn chatbot using Funix. The corresponding app in action is in [](#fig_joke). The only thing Funix-specific in the code is using the `@funix` decorator to change the arrangement of the input and output panels from the default left-right to top-bottom.


    ```{code} python
    :label: code_joke
    :caption: Multiturn chatbot using Funix.

    import IPython     
    from openai import OpenAI
    import funix

    client = OpenAI()

    messages  = []  # list of dicts, dict keys: role, content, system. Maintain the conversation history.

    def __print_messages_html(messages):
        printout = ""
        for message in messages:
            if message["role"] == "user":
                align, left, name = "left", "0%", "You"
            elif message["role"] == "assistant":
                align, left, name = "right", "30%", "ChatGPT"
            printout += f'<div style="position: relative; left: {left}; width: 70%"><b>{name}</b>: {message["content"]}</div>'
        return printout

    @funix.funix(
        direction="column-reverse",
    )
    def ChatGPT_multi_turn(current_message: str)  -> IPython.display.HTML:
        current_message = current_message.strip()
        messages.append({"role": "user", "content": current_message})
        completion = client.chat.completions.create(messages=messages)
        chatgpt_response = completion.choices[0].message.content
        messages.append({"role": "assistant", "content": chatgpt_response})

        return __print_messages_html(messages)
    ```

  
    ```{figure} joke.gif
    :label: fig_joke

    A multi-turn chatbot in Funix in action. Source code in [](#code_joke).
    ```
  
4. Multimodal inputs

   Funix extends the support to `ipywidgets.{Image, Audio, File, Video}` to allow drag-and-drop of multimedia files or push-to-capture audio or video from the computer's microphone or webcam. 
  
    ```{code} python
    :label: code_multimedia
    :caption: A multimodal input demo in Funix built by simply wrapping OpenAI's GPT-4o demo code into a function with an  `ipywidgets.Image` input and a `str` output. The corresponding GUI app is shown in [](#fig_multimedia).
    
    import openai
    import base64
    from ipywidgets import Image
    
    client = openai.OpenAI()

    def image_reader(image: Image) -> str:
      """
      # What's in the image? 

      Drag and drop an image and see what GPT-4o will say about it. 
      """

      # Based on https://platform.openai.com/docs/guides/vision 
      # with only one line of change
      response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
          {
            "role": "user",
            "content": [
               {"type": "text", "text": "What's in this image?"},
               {"type": "image_url",
               "image_url": {
                  "url":f"data:image/png;base64,{base64.b64encode(image).decode()}",
                },
                },
              ],
          }
        ],
      )
      return response.choices[0].message.content
    ```
  
    ```{figure} drag_and_drop.png
    :label: fig_multimedia

    Funix maps a `ipywidgets.{Image, Audio, Video, File}`-type arguments to a drag-and-drop file uploader with push-to-capture ability from the microphone or webcam of the computer. The corresponding source code is in [](#code_multimedia).
    ```

  
5. Vector stripping in bioinformatics. 

   Vector stripping is a routine task in bioinformatics where appendix sequences (called "vectors") padded onto the nucleotide sequences of interest for easy handling or quality control are removed.
   A vector stripping app only involves simple data structures, such as strings, lists of strings, and numeric parameters. This is a sweet spot of Funix. 
   
   Because the bioinformatics part of vector stripping is lengthy, we only show the interface function in [](#code_vector_stripping) and full source code can be found [here](https://github.com/TexteaInc/funix/blob/develop/examples/bioinformatics/vector_strip.py). `pandas.DataFrame`'s are used in both the input and output of this app, allowing biologists to batch process vector stripping by copy-and-pasting their data to Excel or Google Sheets, or uploading/downloading CSV files. 
   
    ```{code} python
    :label: code_vector_stripping
    :caption: The function that is turned into a vector stripping app by Funix. 

    def remove_3_prime_adapter(
        adapter_3_prime: str="TCGTATGCCGTCTTCTGCTT",
        minimal_match_length: int = 8,
        sRNAs: pandas.DataFrame = pandas.DataFrame(
            {
                "sRNAs": [
                    "AAGCTCAGGAGGGATAGCGCCTCGTATGCCGTCTTCTGC",  # shorter than full 3' adapter
                    "AAGCTCAGGAGGGATAGCGCCTCGTATGCCGTCTTCTGCTT",  # full 3' adapter
                    # additional seq after 3' adapter,
                    "AAGCTCAGGAGGGATAGCGCCTCGTATGCCGTCTTCTGCTTCTGAATTAATT",
                    "AAGCTCAGGAGGGATAGCGCCTCGTATG",  # <8 nt io 3' adapter
                    "AAGCTCAGGAGGGATAGCGCCGTATG",  # no match at all
                ]
            }
        ),
        # ) -> pandera.typing.DataFrame[OutputSchema]:
    ) -> pandas.DataFrame:

        ## THE BODY HIDDEN

        return pandas.DataFrame(
            {"original sRNA": sRNAs["sRNAs"], "adapter removed": list(return_seqs)}
        )

    ```

  
    ```{figure} vector_stripping.png
    :label: fig_vector_stripping
    ```

## Conclusion

In this paper, we introduce the philosophy and features of Funix. Funix is motivated by the observations in scientific computing that many apps are straightforward  input-output processes and the apps are meant to be disposable at a large volume. Therefore, Funix' goal is to enable developers, who are experts in their scientific domains but not in frontend development, to build apps by continue doing what they are doing, without code modification or learning anything new. 
To get this goal, Funix leverages the language features of the Python language, including docstrings and keywords, to automatically generate the GUIs for apps and control the behaviors of the app.
Funix tries to minimize reinventing the wheel by being a transcompiler between the Python word and the React world. 
Not only does it expose developers to the limitless resources in the frontend world, but it also minimizes the learning curve. 
Funix is still a very early-stage project. As an open-source project, we welcome feedback and contributions from the community.

## Acknowledgments

Funix is not the first to exploit variable types for automatical UI generation. [Python Fire by Google](https://github.com/google/python-fire) is a Python library that automatically generates command line interfaces (CLIs) from the signatures of Python functions. Funix extends the idea from CLI to GUIs. `interact` in [ipywidgets](https://ipywidgets.readthedocs.io/en/latest/examples/Using%20Interact.html) infers types from default values of keyword arguments and picks widgets accordingly. But it only supports five types/widgets (`bool`, `str`, `int`, `float`, and Dropdown menus) and is not easy to expand the support. Funix supports a lot more types out of the box, requires no modification to the code (vs. calling `ipywidgets.interact`), and exposes the entire frontend ecosystem to users.
