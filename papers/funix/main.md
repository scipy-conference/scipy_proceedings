---
# Ensure that this title is the same as the one in `myst.yml`
title: Funix - vivifying (any) Python functions into GUI apps
abstract: |
  The raise of machine learning (ML) and artificial intelligence (AI), especially the generative AI (GenAI), has brought up the need for wrapping models or algorithms into GUI apps. For example, a large language model (LLM) can be access through a string-to-string GUI app with a textbox as the primary input.
  Most of existing solutions require developers to manually create widgets and link them to arguments/returns of a function individually. This low-level process is labrious and usually intrusive. Funix takes a CSS-like approach to this problem by automatically picking widgets based on the types of the arguments and returns of a function according to the type-to-widget mapping defined in a theme. For example, in Funix' default theme, the Python native types `str`, `bool`, and `Literal` are mapped into an input box, a checkbox, and a set of radio buttons in the MUI library. As a result, an existing Python function can be turned into a GUI app without any code change. Unlike existing solutions that bound the choice of widgets to those provided by them, as a transcompiler, Funix allows a developer to pick any component in the frontend world and customize it in JSON without any knowledge to Javascript/Typescript. Funix further uses the information in docstrings, which are common in Python development, to control the appearance of the GUI. 
---

## Introduction

Presenting a model or an algorithm as a GUI app is a common need in the scientific and engineering community. 
For example, a large language model (LLM) is not accessible to the general public until we wrap it with a chat interface, which consists of a text input and a text output. 
Because most scientists and engineers are not familiar with frontend development which is Javascript/Typescript-centric, there have been many solutions based on Python, one of the most popular programming languages in scientific computing -- especially AI, such as [ipywidgets](),  [Streamlit](), Gradio, Reflex, Dash, and PyWebIO. 
Most of them follow the conventional GUI programming philosophy that a developer needs to manually pick widgets from a widget library and associate them with the arguments and returns of an underlying function, which is usually called the "call back function."

This approach has several drawbacks. **First**, it needs continuous manual work to align the GUI code with the callback function. If the signature of the callback function changes, the GUI code needs to be manually updated. In this sense, the developer is manually maintaining two sets of code that are highly overlapping. **Second**, the GUI configuration is low-level, at individual widgets. It is difficult to reuse the design choices and thus laborious to keep the appearance of the GUI consistent across apps. **Third**, the developer is bounded by the widgets provided by the GUI library. If there is an unsupported datatype, the developer most likely has to give up due to the lack of frotend development knowledge. **Fourth**, they do not leverage the features of the Python language itself to reduce the amount of additional work. For example, it is very common in Python development to use docstrings to annotate arguments of a function. However, most of the existing solutions still requires the developer to manually specify the labels of the widgets. **Last** but not least, all existing solutions require the developer to read their documentations before being able to say "hello, world!" 

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

The app generated from [](#code_hello_world) by the command  `funix hello.py` by Funix.
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

Funix also exploits other features or common practices in Python programming to make building apps more intuitive and efficient. default values, Docstring and multiplage app. 

Funix' architecture: a transcompiler that exposes the frontend world to Python developers.
Funix is more than a GUI generator. As a transcompiler, it produces both the frontend ...

Funix can be really useful in building what we call "disposable apps" -- apps that are built for a short-live purposes, such as collecting human evaluation of an AI model, collecting human labeling of AI training data, exhibit the capabilities of an API, etc. 

In summary, Funix has the following distinctive features compared with many existing solutions:
1. Type-based GUI generation controlled by themes
2. Exposing the frontend world to Python developers
3. Leveraging Python's language features to make building apps more intuitively. In other words, less new stuff from Funix as possible. 

The rest of the paper is organized as follows: 
In [Motivation](#motivation-a-balance-between-simplicity-and-versatility), we discuss the motivation behind Funix and what are the cases suitable and not suitable for Funix. ...

## Motivation: balancing simplicity and versatility for GUI app development

When it comes to app development, there is a trade-off between simplicity and versatility. 
JavaScript/TypeScript-based solutions like React, Angular, and Vue.js are versatile. 
But that versatility is beyond the reach of most scientists and engineers, except frontend/full-stack engineers, and is usually overkilling for most scientific and engineering apps. 

As machine learning researchers ourselves, we notice that a great (if not overwhemling) amount of scientific apps have the following two features: 
1. the underlying logic of an GUI app or a page in a multi-page/tab GUI app is a straightforward input-output process -- thus complex interactivity, such as updating the input options based on existing user input, is not needed;
2. the app is not the goal but a necessary step to the goal -- thus it is not worthy to spend time on building the app.

Most existing solutions do a great job for the first feature above but are still too complicated for the second one, requiring a developer to read their documentations and add some code before an app can be fired up. 
Since versatility is already given up for simplicity in Python-based app building, why not trade it further for more  simplicity?

Funix pushes the simplicity to the extreme. 
In the "hello, world" example ([](#code_hello_world)) above, you get an app without learning anything nor modifying the code. 
While Funix is a button-push solution, it also leaves the room for low-level customization. 

We would also like to argue that the raise of GenAI is simplifying the GUI as natural languages are becoming a prominent interface between humans and computers. Consider text-to-image generation, the app only needs a string input and an image output. In this sense, Funix, and its Python-based peers, will be able to meet a lot of needs, in scientific computing or general, in the future.

Because Funix is designed for quickly firing up apps that model a straightforward input-output process, it has one input panel on the left and an output panel on the right. The underlying function will be called after the user plugs in the arguments and click the "Run" button. The result will be displayed in the output panel.

## Transcompilation from Python types to React components

Like CSS, Funix controls the GUI appearance based on the types of arguments and returns of a function. 
Under the hood, Funix transcompiles the signature of a function into React code. 
Currently, Funix depends on the type hint for every argument or return of a function. In the future, it will support type inference or tracing.

By default, Funix maps the following basic Python types to the following MUI components:

| Python type | MUI component |
|-------------|---------------|
| `str`       | TextField |
| `bool`      | Checkbox  |
| `int`       | TextField |
| `float`     | TextField |
| `Literal`   | RadioGroup if number of elements is below 8; MultiSelect otherwise |
| `range`     | Slider    |
| `List[Literal]`  | An array of Checkboxes if the number of elements is below 8; AutoComplete otherwise |

In particular, we leverage the semantics of `Literal` and `List[Literal]` for single-choice and multiple-choice selections. 

Because Funix is a transcompiler, it benefits from multimedia types defined in popular Python libraries. For example, the following `ipywidgets` and `IPython` types are natively supported by Funix -- although they are mapped to MUI components rather than `ipywidgets` (Jupyter's input widgets) or `IPython` (Jupyter's display system) components:

| Python type | MUI component |
|-------------|---------------|
| `ipywidgets.Password` | TextField with `type="password"` |
| `IPython.display.Image` | Yazawazi: please fill |
| `IPython.display.Video` | Yazawazi: please fill |
| `IPython.display.Audio` | Yazawazi: please fill |

Yazawazi: Please complete the table above. 

Funix also has built-in support to `pandas.DataFrame` and `matplotlib.figure.Figure`, which are mapped to tables and charts. For example, the function below

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

## Expanding or customizing type-to-widget mapping

Because Funix is a transcompiler, a user can link any type to any widget in any frontend library as s/he wishes. 
Besides the default type-to-widget mappings provided by the default theme, Funix allows a user to define a new theme or to override any theme in a local scope on-the-fly. 
In addition to type-based, automatic widget selection, Funix also allows per-variable widget selection. Please refer to the [decorator](#decorator) section for more details.

### Themes

A theme is a nested dictionary in JSON such as the one below which binds `str`, `int`, and `float` to three widgets: 

```json
{
 "name": "test_theme",
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

The `widgets` field defines the binding from Python types to frontend widgets. One can use a module specifier (e.g., `@mui/material/Slider`) in `npm`, the de facto package manager for the frontend world, to refer to a widget and modify its `props` in JSON (e.g., `{"min":0, "max": 100, "step": 0.1}`). For convenience, Funix also provides some shortcut strings for frequently used widget families, such as `inputbox` and `slider[0,100,2]` above. Details for the `widgets` field can be found in the Reference manual of Funix. 

From the example above, we can see that Funix exposes the entire frontend world to Python developers, allowing them to pick any component from any frontend library and configure them without any knowledge of JavaScript/TypeScript or React. In this sense, Funix bridges the Python world with the frontend world and made Python or JSON the surface language for frontend development.

### On-the-fly new type definition

Besides themes, Funix supports introducing new types and binding them to wiedgets on-the-fly.
In the example below, we define a special `str` type called `blackout` that is bounded with `@mui/material/TextField` with `type="password"`. Thus, it will be embodied by a text input box that the content will be hidden while typing unless explicitly toggled to display by the user. 

```python
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

## Building apps Pythonically

Besides the type-to-widget philosophy, Funix leverages the features and common practices in Python to make building apps more intuitive and efficient.

### Default values
The most trivial thing is default values. Python allows default values for keyword arguments. Funix directly uses them as the default values for corresponding widgets. In contrast, Funix' peer solutions require developers to provide the default value the second time in the widget initiation. 

### Docstring-based GUI augmentation

Docstrings are dominantly common in the Python community. 
Information in Docstrings is often needed in the GUI of an app. For example, the annotation of each argument can be displayed as a tooltip when the user hovers over the corresponding widget.
In peer solutions, a developer has to repeat such information in the GUI code.
Funix makes use of the information in Docstrings. It currently only supports Google-style and Numpy-style docstrings. 
The Docstring above the first section (e.g., `Args`) will be rendered as Markdown. Argument annotations in the `Args` section will become the tooltips of the corresponding widgets. Finally, the `Examples` section will become prefilled example values for a user to try out. 

Only information in the `Args` and `Examples` section will be displayed in the GUI.

```{code} python
:label: code_docstring
:caption: An example of a function with a Google-style docstring. The corresponding GUI app is shown in [](#fig_docstring).

def foo(x: str, y: int) -> str:
    """
    ## What happens when you multiply a string with an integer?    

    Try it out below. 

    Args
    ----------
        x: A string that you want to repeat. 
        y: How many times you want to repeat. 

    Examples
    ----------
    >>> foo("hello ", 3)
    "hello hello hello "
    
    """
    return x * y 
```

### Output layout in `print` and `return`

In Funix, by default, the return of a function becomes the content of the output panel. 
A user can control the layout of the output panel returning strings, including and especially f-strings, in the Markdown and HTML syntaxes. 
Markdown and HTML strings must be explicitly specified of the types `IPython.display.Markdown` and `IPython.display.HTML`, respectively. Otherwise, the raw strings will be displayed. Because Python supports multiple returns, you can mix Markdown and HTML strings in the return statement. 

Quite often we need to print out some information before a function reaches its returns.
`print` is a Python built-in function that is frequently used for this purpose.
Funix extends this convenience by directing the output of `print` to the output panel of an app. The printout strings in Markdown or HTML syntax and will be automatically rendered after syntax detection. 
To avoid conflicting with the default behavior of printing to `stdout`, printing to the web needs to be explicitly turned on by a decorator.   

```{code} python
:label: code_print_return
:caption: Use `print` and `return` to control the output layout. The corresponding GUI app is shown in [](#fig_print_return).

from IPython.display import Markdown, HTML
from typing import Tuple
from funix import funix

@funix(print_to_web=True)
def foo(income: int = 200000, tax_rate: float= 0.45) -> Tuple[Markdown, HTML]:
    print (f"## Here is your tax breakdown: ")

    tax_table = """\
      | Item | $$$ | 
      | --- | --- |
      | Income | {income} |
      | Tax | -{tax_rate * income : .3f} |
      | Net Income | {income - tax_rate * income : .3f} |
    """
  
    return tax_table, "If you have any question, contact <a href='http://irs.gov'>IRS</a>."
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

### State keeping, sessionizing, and multi-page apps

A multi-page app is an app of many pages/tabs that share the same variable space. Values of variables set in one page can be accessed in another page. 
Without reinventing the wheel, Funix supports this need by turning a Python class into a multi-page app, making use of the `self` variable for data passing across pages. Each member function becomes a page in the multi-page app. In particular, the constructor (`__init__`) becomes the landing page of such a multi-page app. Because each instance of a class is independent, the multi-page app is sessionized for different connections from browsers. 
In this approach, a Funix developer does not have to learn anything new and it aligns well with the OOP principles that most programmers are already familiar with. 

```{code} python
:label: code_class_multi_page
:caption: A simple multi-page app in Funix leveraging OOP. The corresponding GUI app is shown in [](#fig_stream).

from funix import funix_method, funix_class
from IPython.display import Markdown, HTML

@funix_class()
class A:
    @funix_method(print_to_web=True)
    def __init__(self, a: int):
        """
        Initialize A
        """
        self.a = a
        print(f"`self.a` has been initialized to {self.a}")

    def set(self, b: int) -> Markdown:
        """
        Set the value for `self.a`. 
        """
        old_a  = self.a
        self.a = b
        return (
                "| var | value |\n" 
                "| ----| ------|\n"
            f"| `a` before | {old_a} |\n" 
            f"| `a` after | {self.a} |"
            )

    def get(self) -> HTML:
        return f"The value of `self.a` is <i>{self.a}</i>. "
```

To maintain state without sessionization, e.g., sharing the same variable across all sessions, simple declare a global variable using the `global` keyword.
To force a global variable to be sessionized, i.e., all users have their own copy of the variable, please start Funix with the `-t` flag. 

```{code} python
:label: code_hangman
:caption: A simple Hangman game in Funix that uses the `global` keyword to maintain the state. This solution is much shorter than using peer solutions, such as in [Gradio](https://www.gradio.app/guides/state-in-blocks). The corresponding GUI app is shown in [](#fig_hangman).

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

## Miscellaneous features 

### History 

### Jupyter support 

### Decorator

Manually pick widgets per-variable

Rate limiting 

### Reactive and continous update

## Showcases

1. Wordle
2. Dall-E
3. ChatGPT multi-turn
4. Continous Sine functions
5. Drag and drop 

## Future work 

## Acknowledgments
Funix is not the first to exploit variable types for automatical UI generation. [Python Fire by Google](https://github.com/google/python-fire) is a Python library that automatically generates command line interfaces (CLIs) from the signatures of Python functions. Funix extends the idea from CLI to GUIs. `interact` in [ipywidgets](https://ipywidgets.readthedocs.io/en/latest/examples/Using%20Interact.html) infers types from default values of keyword arguments and picks widgets accordingly. But it only supports five types/widgets (`bool`, `str`, `int`, `float`, and Dropdown menus) and is not easy to expand the support. Funix supports a lot more types out of the box, requires no modification to the code (vs. calling `ipywidgets.interact`), and exposes the entire frontend ecosystem to users.