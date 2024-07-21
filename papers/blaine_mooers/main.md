---
# Voice Computing with Python in Jupyter Notebooks
title: Voice Computing with Python in Jupyter Notebooks
abstract: |
    Jupyter is a popular platform for writing interactive computational narratives that contain computer code and its output interleaved with prose that describes the code and the output. It is possible to use one's voice to interact with Jupyter notebooks. This capability improves access to those with impaired use of their hands. Voice computing also increases the productivity of workers who are tired of typing, and increases the productivity of those workers who speak faster than they can type. Voice computing can be divided into three activities: speech-to-text, speech-to-command, and speech-to-code. We will provide examples of the first two activities with the Voice-In Plus plugin for Google Chrome and Microsoft Edge. To support the editing of Markdown and code cells in Jupyter noteobooks, we provide several libraries of voice commands at MooersLab on GitHub.
---


## Introduction

Voice computing includes speech-to-text, speech-to-commands, and speech-to-code. 
These activities enable you to use your voice to generate prose, operate your computer, and write computer code.
This use of your voice can partially replace the use of the keyboard when tired of typing, when suffering from repetitive stress injuries, or both.
We found that we could be productive within an hour with the Voice In Plus plugin for Google Chrome and Microsoft Edge.
This plugin is easy to install, provides accurate dictation, and is easy to modify to correct wrong word insertations with text replacements.

We discovered that we could map the words to be replaced, what we call *voice triggers*, to equations set in LaTeX and to code snippets that span one-to-many lines.
These *voice-triggered snippets* are analogous to traditional tab-triggered snippets supported by most text editors.
(A tab trigger is a placeholder word that is replaced with the corresponding code when the tab key is pressed after entering the tab trigger.)
The existing extensions for code snippets in Jupyter do not support tab triggers.
Instead, we found that Voice In Plus can be used to insert voice-triggered snippets into code and Markdown cells in Jupyter notebooks.
These snippets require customizing to the problem at hand via the keyboard, but their insertion by voice command still saves time.

To facilitate the use of voice commands in Jupyter notebooks, we have developed libraries of voice-triggered snippets for use in Markdown or code cells with the *Voice-In Plus* plugin.
We are building on our experience with tab-triggered code snippets in text editors [@Mooers2021TemplatesForWritingPyMOLScripts] and domain-specific code snippet libraries for Jupyter [@Mooers2021APyMOLSnippetLibraryForJupyterToBoostResearcherProductivity].
We have made libraries of these voice-triggered snippets for several of the popular modules of the scientific computing stack for Python.
These voice-triggered snippets are another tool for software engineering that complement existing tools for enhancing productivity.

## Methods and Materials

### Hardware
We used a 2018 15-inch MacBook Pro laptop computer.
It had 32 gigabytes of RAM and one Radeon Pro 560X 4 GB GPU.
We used the laptop's built-in microphone to record dictation while sitting or standing up to 20 feet (ca. 6 m) away from the computer.

### Installation of Voice In Plus
We used the *Voice In* plugin provided by Dictanote Inc.
First, we installed the *Voice In* plugin by navigating to the [Plugin In page](https://chromewebstore.google.com/detail/voice-in-speech-to-text-d) in the Google Chrome Web Store on the World Wide Web.
Seond, the [Microsoft Edge Add-ons web site](https://microsoftedge.microsoft.com/addons/Microsoft-Edge-Extensions-Home) was accessed to install the plugin in Microsoft Edge.

An internet connection was required to use *Voice In* because Dictanote tracks the websites visited and whether the plugin worked on those websites.
*Voice In* uses the browser's built-in the Speech-to-Text software to transcribe speech into text.
Because no remote servers were used for the transcription, the transcription process was essentially instant and able to keep up the dictation of multiple paragraphs.
Dictanote does not store the audio or the transcripts.

After activating the plugin, we customized it by selecting a dictation language from a pull-down menu.
We selected **English (United States)** from the 15 dialects of English. 
The English variants include dialects from Western Europe, Africa, South Asia, and Southeast Asia; many languages other than English are also supported.

Next, we set a keyboard shortcut for activating the plugin.
We selected command-L on our Mac because this shortcut was not already in use.
A list of Mac shortcuts can be found [here](https://support.apple.com/en-us/102650).
The second customization was the limit of the customizations that we could make in the free version of *Voice In*.

The use of customized text replacements is available in *Voice In Plus*.
*Voice In Plus* is activated by purchasing a $39 annual subscription through the **Subscription** submenu in the *Voice In Settings* sidebar of the *Voice In Options* web page.
Monthly and lifetime subscription options are available.
Only one subscription was required to use *Voice In Plus* in both web browsers.
The library is synched between the browsers.

After a subscription was purchased, we obtained access to the **Custom Commands** in the *Voice In Settings* sidebar.
We used the **Add Command** button to enter a voice-trigger to start a new sentence when writing one sentence per line {ref}`fig:newSentence`.
The custom command for `new paragraph` can include a `period` followed by two `new line` commands, which works well when writing with blank lines between paragraphs and without indentation at the beginning of each paragraph.
The phrases making up these voice triggers were no longer be available during dictation because they will be used to trigger text replacements.

:::{figure} ./images/VoiceInNewSentence.png
:label: fig:newSentence
:width: 50%
Entering a single voice trigger and the corresponding command in Voice In Plus.
:::

We used the **Bulk Add** button to upload multiple commands from a two-column CSV file with commas as the field sepearator.
The file contents were selected and pasted in a text box that is opened upon clicking on the **Bulk Add** button.
The voice-triggers reside in the left column, and the text replacements reside in the right column.
Any capitalization in the voice trigger was ignored by the software.

Multiline text replacements had to be enclosed in double quotes.
Internal double quotes were replaced with single quotes.
It was not possible to use a backspace to escape internal pre-existing double quotation marks.

The formatting of the text replacement was controlled by inserting built-in *Voice In Plus* commands enclosed in angle brackets.
For example, the built-in **open** command enables the opening of a webpage with the provided URL (e.g. `open SciPy 2024,<open:https://www.scipy2024.scipy.org/schedule>`).

The transcription of the spoken words appear momentarily in a transitory transcript that hovers about the text box.
We mapped the misinterpreted words to the desired text replacement.
For example, we added the mapping of `open syfy` to `<open:https://www.scipy2024.scipy.org/schedule>` to open the webpage for the schedule of SciPy 2024 when we said, "open SciPy 2024".

The **Export** button opened a text box with the custom commands in CSV file format.
All the contents of the text box were selected, copied, and pasted into a local CSV file using either the text editor TextMate or Emacs version 29.3.
The **Synch** button was used to synchronize devices.

A GUI shows all the voice triggers and their text replacements immediately below the row of buttons just mentioned above.
Each row in the GUI has an edit icon and a delete icon.
The edit icon opens a pop-up menu similar to the pop-up menu invoked by the **Add Command** button.

### Construction of the snippet libraries
Some of our voice snippets had already been used for a year to compose prose using dictation.
These snippets are in modular CSV files to ease their selective use.
The contents of these files can be copied and pasted into the `bulk add` text area of the Voice In Plus configuration GUI.

### Construction of interactive quizzes
We developed interactive quizzes to aid the mastery of the *Voice In Plus* syntax.
We wrote the quiz as a Python script that can run in the terminal or in Jupyter notebooks.
We stored each question, answer, and explanation triple in a tuple because tuples are immutable. 
We stored the tuples in a list because lists are sortable.
The quiz randomizes the order of the questions upon restart, ensuring a fresh and challenging experience every time.
When you fail to answer a question correctly, the quiz provides feedback, giving you the opportunity to learn from your mistakes and improve.
Recycling the wrongly answered questions builds the recall of the correct answers.
The number of questions in a quiz was limited to 40 to avoid exhausting the user.

A function writes out the quiz to a PDF file upon completion of the quiz or upon early exit from the quiz.
The user can take a paper version of the quiz while away from the computer.

### Availability of the libraries and quizzes
We tested the libraries using Jupyter Lab version 4.2 and Python 3.12 installed from MacPorts.
All libraries are made available at MooersLab on GitHub for download.

## Results

First, we describe the contents of the snippet libraries and the kinds of problems they can solve.
We group the libraries into several categories to simplify their explanation.
Second, we describe the deployment of the snippet libraries for Voice In Plus (VIP), an automated speech recognition plugin for the web browsers Google Chrome and Microsoft Edge.
The Voice In Plus plugin has a gentle learning curve.

### Composition of the libraries

We present the libraries we crafted to streamline using Jupyter with voice commands.
Our descriptions of these libraries illustrate how voice-triggered snippets work with automated speech recognition software.
Developers can leverage our libraries in two ways: by enhancing them with their unique commands or by using them as blueprints to create their libraries from scratch.

The libraries are made available in a modular format so the user can select the most valuable commands for their workflow.
In general, our libraries are broad in scope, so they meet the needs of most users.
Several libraries are domain-specific.
These domain-specific libraries serve as a catalyst for the creation of libraries tailored to other fields, sparking innovation, and expanding the reach of voice-triggered snippets.

We divided the contents of the libraries into two categories.
One subset of libraries supports dictating about science in the Markdown cells of Jupyter notebooks, while the other subset supports writing scientific Python in code cells.
These voice-triggered snippets can also be applied to Markdown and code cells in Colab Notebooks.
While some code, such as IPython line and cell magics, is specific to Jupyter, most of the voice-triggered snippets can be used in Markdown files and Python scripts being edited in Jupyter Lab.
Likewise, these snippets can be used in other browser-hosted text editors such as the new web version of Visual Studio Code.
This is because Voice In Plus works in most text areas of web browsers.
These means that ample text area like that found in Write Honey site can be used to draft documents and Python scripts with the aid of Voice In Plus snippets and then be directly edited by an advanced text editor by using GhostText to connect the web-based text area to a text editor.
Alternately, the draft document or script can be copied and saved to a file that is imported into Jupyter Lab for furtther editing.

### Libraries for Markdown cells

These libraries contain a short phrase paired with its replacement: another phrase or a chunk of computer code.
In analogy to a tab trigger in text editors, we call the first short phrase a voice trigger.
Tab triggers are initiated by typing, followed by the tab key, which inserts the corresponding snippet of code.
Some text editors can autocomplete the tab-trigger name, so these text editors require two tab key entries. 
The first tab auto-completes the tab-trigger name.
Then, the second tab leads to the insertion of the corresponding code.
This two-step process of tab triggers empowers users with the flexibility to select a different tab trigger before inserting the code, enhancing the customization potential of text editors.
It is important to note that voice triggers, while efficient, do not allow for the revision of the voice trigger. 
In the event of an incorrect code insertion, the undo command must be used, underscoring the need for caution when selecting voice triggers.

The most straightforward text replacements involved the replacement of English contractions with their expansions {ref}`fig:contractions`.
Science writers do not use English contractions in formal writing for many reasons.
Many automatic speech recognition software packages will default to using contractions because the software's audience is people who write informally for social media, where English contractions are acceptable.
By adding the library that maps contractions to their expansions, the expansion will be inserted whenever the contraction is used otherwise.
This automated replacement of English contractions saves time during the editing process.

:::{figure} ./images/VoiceInCustomCommandsContractions.png
:alt: contractions
:width: 60%
:align: center
:label: fig:contractions

Webpage of custom commands. The buttons are used to edit existing commands and add new commands. Three English contractions and their expansions are shown.
:::

We grouped the voice triggers by the first word of the voice trigger {ref}`table:commands`.
This first word is often a verb.
For instance, the word `expand' is a practical and useful command in our library. 
This command expands acronyms, making it easier for users to understand the content.

One's memory of the words represented by the letters in the acronym often needs to be more accurate.
The library ensures that one uses the correct expansion of the acronym.
By providing instant access to the correct acronym expansions, the library significantly reduces the time that would otherwise be spent on manual lookups, allowing you to focus on your work.
We have included acronyms widely used in Python programming, scientific computing, data science, statistics, machine learning, and Bayesian data analysis.
These commands are found in domain-specific libraries, so users can select which voice triggers to add to their private collection of voice commands.

:::{table} Examples of voice commands with the prefix in bold that is used to group commands.
:label: table:commands
:align: center

| Voice commands                                                                            |
|:------------------------------------------------------------------------------------------|
| **expand** acronyms                                                                       |
| **the book** title                                                                        |
| **email** colleagues (i.e., inserts list of e-mail addresses)                                    |
| **insert** code blocks (including equations in LaTeX)                                     |
| **list** (e.g., font sizes in \LaTeX, steps in a protocol, members of a committee, etc.)  |
| **open** webpage (e.g., open pytexas, open google scholar)                                |
| **display** insert equation in display mode (e.g., display electron density equation)   |
| **display with terms** above plus list of terms and their definitions  (e.g., display electron density equation)   |
| **inline** equation in-line (e.g., inline information entropy equation)                   |
| **citekey** insert corresponding citekey (e.g., citekey Wilson 1942)  |
::::

:::{figure} ./images/DisplayElectronDensityEquation.png
:alt: display
:width: 100%
:align: center
:label: fig:displayeq

Three snapshots from a Zoom video of using the voice-trigger *display electron density equation* in a Markdown cell in a Jupyter notebook. A. The Zoom transcript showing the spoken voice trigger. B. The text replacement in the form of a math equation written in LaTeX in display mode in the Markdown cell. C. The rendered Markdown cell. The green and black tab on the right of each panel indicates that the Voice In plugin is active and listening for speech.
:::

Another example of a verb starting a voice trigger is the command `display  <equation name>`.
This command is used in Markdown cells to insert equations in the display mode of LaTeX in Markdown cells.
For instance, the voice trigger `display the electron density equation` is a testament to the convenience of our system, as shown in the transcript of a Zoom video {ref}`fig:displayeq`. 
The image in the middle shows the text replacement as a LaTeX equation in the display mode.
This image is followed by the resulting Markdown cell after rendering by running the cell.

Likewise, the command `inline <equation name>` is used to insert equations in prose sections in Markdown cells.
We've introduced practical voice-triggered snippet libraries, specifically designed for equations commonly used in Bayesian data analysis and structural biology.
While the development of libraries are still in progress, they serve as flexible templates that can be adapted for any specific domain.

Some voice triggers start with a noun.
For example, the voice trigger `URL` is used to insert URLs for essential websites.
Another example involves using the noun `list` in the voice trigger, as in `list matplotlib color codes`, to generate a list of the color codes used in Matplotlib plots.
These voice triggers, such as 'URL' or 'list matplotlib color codes ', provide instant access to essential information, saving you the time and effort of manual searches.

The markup language code is inserted using the verb *insert*, followed by the markup language name and the name of the code.
For example, the command `insert markdown itemized list` will insert five vertically aligned dashes to start an itemized list.
The command `insert latex itemized list` will insert the corresponding code for an itemized list in LaTeX.

We have developed a library specifically for the flavor of [Markdown](https://github.com/MooersLab/markdown-jupyter-voice-in/blob/main/markdown.csv) utilized in Jupyter notebooks.
This library is used to insert the appropriate Markdown code in Markdown cells.

We have included a [library for LaTeX](https://github.com/MooersLab/latex-voice-in) because Jupyter Lab can edit text files.
According to the above rule, we should use `insert latex equation` to insert the code for the equation environment.
However, we broke this convention by omitting the word `latex` to make the commands more convenient because this is the default typesetting language we use daily.
We were already comfortable with using the abbreviated commands.

We have not figured out how to use voice commands to advance the cursor in a single step to sites where edits should be made, analogous to tab stops in conventional snippets.
Instead, the built-in voice commands can move the cursor forward or backward and select replacement words.
We included the markup associated with the yasnippet snippet libraries to serve as a benchmark for users to recognize the sites that should be considered for modification to customize the snippet for their purpose.


### Libraries for code cells
The `insert` command in code-cell libraries is a powerful tool that allows the seamlessly insert of chunks of Python code, enhancing your data science workflow.
We avoid making voice commands for small code fragments that might fit on a single line.
An exception to this was the inclusion of a collection of one-liners that are in the form of several kinds of comprehensions.
As mentioned earlier, we have developed practical chunks of code that are designed to perform specific functions, making your data analysis tasks easier and more efficient.
These chunks of code could be functions or lines of code that produce an output in the form of a table, plot, or analysis.
The idea was to fill a code cell with all the code required to produce the desired output.
Unfortunately, the user will still have to use their voice or computer mouse to move the cursor back over the code chunk and customize portions of the code chunk as required for the task at hand.

While self-contained examples that utilize generated data can illustrate concepts, these examples are frustrating for beginners who need to read actual data and would like to apply the code example to their problem.
Reading appropriately cleaned data is a common task in data science and a common barrier to applying Jupyter notebooks to scientific problems.
Our data wrangling library provides code fragments that directly import various file types, easing the tedious task of data import and allowing focus on downstream utilization and analysis.

After the data is input, it needs to be displayed in a tabular format for inspection to check that it was properly imported and to carry out basic summarization statistics by column and row.
After the data are verified as being correctly imported, it is often necessary to explore them by plotting them to detect relationships between a model's parameters and the output.
Our focus on the versatile matplotlib library, which generates a wide variety of plots, is designed to inspire creativity in data visualization and analysis.
Our code fragments cover the most commonly used plots, such as scatter plots, bar graphs (including horizontal bar graphs), kernel density fitted distributions, heat Maps, pie charts, and contour plots.
We include a variety of examples for the formatting of the tick marks and axis labels as well as the keys and the form of the lines so users can use this information as templates to generate plots for their own purposes.
The generation of plots with lines of different shapes, whether solid, dashed, dotted, or combinations thereof, is essential because plots generated with just color are vulnerable to having their information compromised when printed in grayscale.
Although we provide some examples from higher-order plotting programs like Seaborn, we focused on matplotlib because most of the other plotting programs, with the exception of the interactive plotting programs, are built on top of it.

We also support the import of external images.
This is often overlooked, but these externally derived images are often essential parts of the story the Jupyter notebook is telling.

### Jupyter specific library

We provide a [library](https://github.com/MooersLab/jupyter-voice-in/blob/main/jupyter.csv) of 85 cell and line magics that facilitate the Jupyter notebook's interaction with the rest of the operating system.
Our cell magics, easily identifiable by their cell magic prefix, and line magics, with the straightforward line magic prefix, are designed to make the Jupyter notebook experience more intuitive.
For example, the voice command *line majic run* insert `%run`; it is used to run script files after supplying the name of the script file as also shown in {fig}`quiz`.

### Interactive quiz

We developed a [quiz](https://github.com/MooersLab/voice-in-basics-quiz) to improve recall of the voice commands.
These quizzes, designed for your convenience, are interactive and can be run in the terminal or in Jupyter notebooks {ref}`fig:quiz`.
The latter can store a record of one's performance on a quiz.

:::{figure} ./images/runningQuiz.png
:label: fig:quiz
:width: 130% 
An example of an interactive session with a quiz in a Jupyter notebook. The code for running the quiz was inserted into the code cell with the voice command `run voice in quiz`. The quiz covers a range of voice commands, including [specific voice commands covered in the quiz].
:::

To build long-term recall of the commands, one must take the quiz five or more times on alternate days, according to the principles of spaced repetition learning.
These principles were developed by the German psychologist Hermann Ebbinghaus in the last part of the 19th Century.
They have been validated several times by other researchers.
Space repetition learning is one of the most firmly established results of research into human psychology.

Most people need more discipline to carry out this kind of learning because they have to schedule the time to do the follow-up sessions.
Instead, most people will find it more convenient to take these quizzes several times in 20 minutes before they spend many hours utilizing the commands.
If that use occurs on subsequent days, then recall will be reinforced, and retaking the quiz may not be necessary.


### Limitations on using Voice In Plus

The plugin operates in text areas on thousands of web pages.
These text areas include those of web-based email packages and online sites that support distraction-free writing like [Write Honey](https://app.writehoney.com).
These text areas also include the Markdown and code cells of Jupyter notebooks and other web-based computational notebooks.
Voice In also works in plain text documents opened in Jupyter Lab for online writing.
It also works in the web-based version of [VS Code](https://vscode.dev/).
Voice In will not work in desktop applications that support the editing of Jupyter notebooks, such as the *JupyterLab", the *nteract*, and external text editors, such as *VS Code*, that support the editing of Jupyter notebooks.
*Voice In Plus* is limited to web browsers, whereas other automated speech recognition  software can also operate in the terminal and at the command prompt in GUI-driven applications. 

It is very accurate with a word error rate that is well below 10\%.
Like all other dictation software, word error rate depends on the quality of the microphone using used.
*Voice-In Plus* can pick out words from among background ambient noise such as load ventilation systems, traffic, and outdoor bird songs.

The language model is quite robust in that dictation can be performed without an external microphone.
We found no reduction in word error rate when using a high-quality Yeti external microphone.
Our experience might a reflection of our high-end hardware and may not transfer to other low-end computers.

Because of the way*Voice-In Plus*is set up to utilize the Speed-to-Text feature of the Google API, there is not much of a latency issue.
The spoken words' transcriptions occur nearly in real-time; there is only a minor lag.
*Voice In Plus* will listen for words for 3 minutes before automatically shutting down.
*Voice In Plus* can generally keep up with dictation occurring at a moderate pace for at least several paragraphs, whereas competing dictation software packages tend to quit after one paragraph.
The program tends to hallucinate only when the dictation has occurred at high speed because the transcription has fallen behind.
You have to pay attention to the progress of the transcription if you want all of your spoken words captured.

If the transcription halts, it is best to deactivate the plugin, activate it, and resume the dictation.
Great care must be taken to pronounce the first word of the sentence loudly so that it will be recorded; otherwise, this first word will likely not be recorded.
This problem of omitted words is most acute when there has been a pause in the dictation.

The software does not automatically insert punctuation marks.
You have to vocalize the name of the punctuation mark where it is required.
You also have to utilize the built-in new-line command to start new lines.
We have combined the period command with the new line command to create a new command with the voice trigger of `new sentence`.

You have to develop the habit of using this command if you like to write one sentence per line.
This latter form of writing is very useful for first drafts because it greatly eases the shuffling of sentences in a text editor during rewriting.
This form of writing is also very compatible with Version control systems like git because the changes can be tracked by line number.

The practical limit of on the number of commands it set by the trouble you are willing to tolerate in scrolling up and down the list of commands.
We had an easy time scrolling through a library of about 7,000 commands, and a hard time with a library of about 19,000 commands.
Bulk deletion of selected commands required the assistance from the User support at Dictanote Inc.
They removed our bloated library, and we used the bulk add button to upload a smaller version of our library.

## Discussion

The following discussion points are crucial for understanding the implementation of the ASR libraries that we have described.
We limit the discussion to the software that we have presented above.

### Independence from breaking changes in Jupyter

The Jupyter project lacks built-in support for libraries of code snippets.
The development of third-party extensions is a necessity to support code snippets.
Unfortunately, changes in the core of Jupyter occasionally break these extensions.
Users have to create Python environments for older versions of Jupyter to work with their outdated snippet extension while missing out on the new features of Jupyter.
An obvious solution to this problem would be for the Jupyter developers to incorporate one of the snippet extensions into the base distribution of Jupyter to ensure that at least one form of support for snippets is always available.
Using voice-triggered snippets external to Jupyter side steps difficulty with breaking changes to Jupyter.


### Voice-triggered snippets can complement AI-assisted voice computing

The use of voice-triggered snippets requires knowledge of the code that you want to insert.
The is the cost that had to be paid to gain access to quickly inserted code snippets that work.
In contrast, AI assistants can find code that you do not know about to solve the problem described in your prompt.
From personal experence, the retrieval of the correct code can take multiple iterations of refining the prompt.
Expert users will find the correct code in several minutes while beginners may take much longer.
An area of future research is to use AI assistants that have libraries indexed on snipppet libraries to retrieve the correct voice-triggered snippet.

### Automated speech recogination extensions for Jupyter lab

We found three extensions developed for Jupyter Lab that enable speech recognition in Jupyter notebooks.
The first, [jupyterlab-voice-control](https://github.com/krassowski/jupyterlab-voice-control), supports custom commands and relies on the browser's language model.
This extension is experimental and not maintained; it does not work with Jupyter 4.2.
The second extension, [jupyter-voice-comments](https://github.com/Banpan-Jupyter-Extensions/jupyter-voice-comments),  relies on the DaVinci large language model to make comments in Markdown cells and request code fragments.
This program requires clicking on a microphone icon repeatedly, which makes the user vulnerable to repetitive stress injuries.
The third extension is [jupyter-voicepilot](https://github.com/JovanVeljanoski/jupyter-voicepilot).
Although the extension's name suggests it uses GitHub's Copilot, it uses whisper-1 and ChatGPT3.
This extension requires an API key for ChatGPT3.
The robustness of our approach is that the *Voice-In Plus* should work in all versions of Jupyter Lab and Jupyter Notebook.

### Coping with the imperfections of the language model

One aspect of speech-to-text that it is important to bring up is persistent errors in translation.
These persistent errors may be due to the language model having difficulties interpreting your speech.
For example, the language model often misinterprets the word  *write* as *right*'.
Likewise, the letter R is frequently returned as *are* or *our*'.
I have had trouble enunciating the letters *w* and *r* clearly since I was a child.
The remedy for the situations is to map the misinterpreted phrase to the intended phrase.

This remedy might be the best that can be done in for those users who have a heavy accent and are from a country that is not represented by the selection of English dialects.
People originating from Eastern Europe, Middle East, and northeast Asia fall into this category.
Users in this situation may have to add several hundred to several thousand text replacements.
As their customized library of text replacements grows, the frequency of further wrong word insertions should go down exponentially.


### Future directions

One future direction is to build out the libraries of voice-triggered snippets that have been developed.
Another direction includes the development of voice stops analogous to tab stops in code snippets for advanced text editors.
These voice stops would advance the cursor to all sites that should be considered for editing to customize the code snippet for the problem.
The other related advance would be the mirroring of the parameter values at identical voice stops.
