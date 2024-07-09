---
# Voice Computing with Python in Jupyter Notebooks
title: Voice Computing with Python in Jupyter Notebooks
abstract: |
    Jupyter is a popular platform for writing literate programming documents that contain computer code and its output interleaved with prose that describes the code and the output. It is possible to use one's voice to interact with Jupyter notebooks. This capability opens up access to those with impaired use of their hands. Voice computing also increases the productivity of workers who are tired of typing, and increases the productivity of those workers who speak faster than they can type. Voice computing can be divided into three activities: speech-to-text, speech-to-command, and speech-to-code. Several automated speech recognition software packages operate on Jupyter notebooks and support these three activities. We will provide examples of the first two activities. Several software tools at MooersLab on GitHub facilitate the use of voice computing software in Jupyter.
---


## Introduction


Jupyter notebooks provide a highly interactive computing environment where users run Markdown and code cells to yield almost instant results.
This form of interactive computing provides the instant gratification of seeing the results of the cells' execution; this might be why Jupyter is so popular for data analysis [@Perkel2021TenComputerCodesThatTransformedScience].
The most popular modality for interacting with the Jupyter notebooks is to use the keyboard and the computer mouse.
However, there are opportunities to generate prose and code using one's voice instead of one's hands.
Users can enhance their prose generation with their voice to boost their productivity and give their hands a rest when fatigued from typing.
In other words, most users can use their voices to complement their keyboard use.
For example, dictating prose in Markdown cells is an obvious application of voice computing in Jupyter.
The ease of generating prose via speech can promote more complete descriptions of the computations executed in adjacent code cells.

Some Speech-to-text software also supports mapping a word or phrase to a text replacement; there are many ways of exploiting text replacements in Markdown and code cells.
For Markdown cells, we have mapped the English contractions to their expansions, so whenever we say a contraction, the expansion automatically replaces the contraction. 
This automation significantly reduces the need for manual editing, saving you valuable time and effort. By leveraging voice commands and text replacements, you can streamline your workflow and focus on the more critical aspects of your work.
Another class of text replacements is the expansion of acronyms into the phrase they represent.
The BibTeX cite keys for standard references can also be mapped to a command like `cite key for scipy`.
Equations type set in LaTeX for rendering with MathJaX can be mapped to commands like `inline pythargeous theorem` and `display electron density equation`, depending on whether the equation is to be *in-line* in a sentence or centered in *display-mode*.
We also mapped voice commands to tables, templates, and software licenses.
For Jupyter code cells, we mapped voice commands to chunks of code of various sizes.
In analogy to tab triggers with conventional tab-triggered snippets in advanced text editors, we call these voice commands that trigger a text replacement *voice triggers*.

To facilitate voice commands in Jupyter notebook cells, we have developed sets of voice-triggered snippets for use in Markdown or code cells.
We are building on our prior experience with tab-triggered code snippets in text editors [@Mooers2021TemplatesForWritingPyMOLScripts] and domain-specific code snippet libraries for Jupyter [@Mooers2021APyMOLSnippetLibraryForJupyterToBoostResearcherProductivity].
We have made libraries of these voice-triggered snippets for several of the popular modules of the scientific computing stack for Python.
Although the Jupyter environment supports polyglot programming, we have restricted our focus to Python and Markdown.
While some code snippets are one-liners, most code snippets span many lines and perform a complete task, such as generating a plot from a data file.
These libraries provide code that is known to work, unlike the situation with chatbots, which do not always return working code.
The high reliability of our code snippet libraries assures you that the code you use is proven to work. 
This trustworthiness suggests that the demand for them will persist through the current AI hype cycle [@Dedehayir2016TheHypeCycleModelAReviewAndFutureDirections], providing you with a stable and dependable tool for your Jupyter notebook projects.

## Methods and Materials

### Construction of the snippet libraries
Some of our voice snippets had already been used for a year to compose prose using dictation.
These snippets are in modular files to ease their selective use.
The Voice-In Plus software accepts commands in a CSV file.
The contents of these files can be copied and pasted into the `bulk add` text area of the Voice In Plus configuration GUI.

### Construction of interactive quizzes
We developed interactive quizzes to aid the mastery of the VIP syntax.
We wrote the quiz as a Python script that can run in the terminal or in Jupyter notebooks.
We stored each question, answer, and explanation in a tuple because tuples are immutable. 
We stored the tuples in a list because lists are sortable.
The quiz randomizes the order of the questions upon restart, ensuring a fresh and challenging experience every time.

Our quiz is not just a test, it's a learning tool. When you fail to answer a question correctly, the quiz provides feedback, giving you the opportunity to learn from your mistakes and improve.
Recycling the wrongly answered questions builds the recall of the correct answers.
The number of questions in a quiz is limited to 40 to avoid exhausting the user.

A function writes out the quiz to a PDF file upon completion of the quiz or upon early exiting the quiz.
This PDF enables the printing of a quiz so a user can take the paper version while away from the computer.

### Availability of the libraries and quizzes
We tested the libraries using Jupyter Lab version 4.2 and Python 3.12 installed from MacPorts.
All libraries are made available at MooersLab on GitHub for download.

## Results

First, we describe the contents of the snippet libraries and the kinds of problems they can solve.
We group the libraries into several categories to simplify their explanation.
Second, we describe the deployment of the snippet libraries for Voice In Plus (VIP), an automated speech recognition plugin for the web browsers Google Chrome and Microsoft Edge.
The Voice In Plus plugin has a gentle learning curve.
The plugin requires an Internet connection to run.
The results section that follows describes the libraries in the same order.

### Composition of the libraries

We present the libraries we crafted to streamline using Jupyter with voice commands.
Our descriptions of these libraries illustrate how voice-triggered snippets work with automated speech recognition software.
Developers can leverage our libraries in two ways: by enhancing them with their unique commands or by using them as blueprints to create their libraries from scratch.

The libraries are made available in a modular format so the user can select the most valuable commands for their workflow.
In general, our libraries are broad in scope, so they meet the needs of most users.
Several libraries are domain-specific.
These domain-specific libraries serve as a catalyst for the creation of libraries tailored to other fields, sparking innovation and expanding the reach of voice-triggered snippets.

We divided the contents of the libraries into two categories.
One subset of libraries supports dictating about science in the Markdown cells of Jupyter notebooks, while the other subset supports writing scientific Python in code cells.
While some code, such as line and cell magic, is specific to Jupyter, most of the voice-triggered snippets can be used in Markdown files and Python scripts being edited in Jupyter Lab.

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
| **email** dsw (i.e., inserts list of e-mail addresses)                                    |
| **insert** code blocks (including equations in LaTeX)                                     |
| **list** (e.g., font sizes in \LaTeX, steps in a protocol, members of a committee, etc.)  |
| **open** webpage (e.g., open pytexas, open google scholar)                                |
g| **display** equation in display mode (e.g., display electron density equation)            |
| **inline** equation in-line (e.g., inline information entropy equation)                   |
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
Instead, most people will find it more convenient to take these quizzes several times in a half hour before they spend many hours utilizing the commands.
If that use occurs on subsequent days, then recall of the alphabet will be reinforced, and retaking the quiz may not be necessary.


### Voice In Plus

Voice In, a unique plugin for Google Chrome and Microsoft Edge web browsers, that uses Google API to provide a dictation and voice recognition experience.
The plugin operates in most text areas of web pages.
These text areas include those of web-based email packages and online sites that support distraction-free writing.
These text areas also include the Markdown and code cells of Jupyter notebooks.
Voice In also works in plain text documents opened in Jupyter Lab for online writing.
Voice In will not work in standalone applications that support the editing of Jupyter notebooks, such as the Jupyter Lab app, the nteract app, and external text editors, such as VS Code, that support the editing of Jupyter notebooks.

After Voice-In Plus is activated, it will listen for words for 3 minutes before automatically shutting down.
It is very accurate with a word error rate that is well below 10\%.
It can pick out words, such as traffic or bird songs, despite background ambient noise.
The language model is quite robust in that dictation can be performed without an external microphone.
For example, the built-in microphone available in the MacBook Pro laptop computer is sufficient.
In contrast, other VSR software requires high-quality external microphones.
The need to use an external microphone imposes a motivational barrier.

Because of the way the system is set up to utilize the Google API, there is not much of a latency issue.
The spoken words' transcriptions occur nearly in real-time; there is only a minor lag.
The program can generally keep up with dictation occurring at a moderate pace for at least several paragraphs, whereas competing systems tend to quit after one paragraph.
The program tends to hallucinate only when the dictation has occurred at high speed because the transcribing falls behind.
As a result, the user has to pay attention to the progress of the transcription.
If the transcription halts, it is best to deactivate the plugin, activate it, and resume the dictation.
Great care must be taken to pronounce the first word of the sentence loudly so that it will be recorded; otherwise, this first word will likely not be recorded.
This problem is most acute when there has been a pause in the dictation.

The software does not automatically insert punctuation marks.
The user has to vocalize the name of the punctuation mark that they want inserted.
These also have to utilize a built-in new-line command to start new lines.
The user has to develop the habit of using this command if they write one sentence per line.
This latter form of writing is very useful for first drafts because it greatly eases the shuffling of sentence order during rewriting.
This form of writing is also very compatible with Virgin control systems like git because the changes can be tracked more easily by line number.

The program can sometimes be unresponsive.
In these situations, the plugin can be turned on and then again.
This act will restore normal behavior.

The associated configuration GUI for VIP allows customization of several settings, making the dictation experience personalized.
The first setting to be set is the language that will be used during dictation.
There is support for several foreign languages and different dialects of English.
The user can also configure a keyboard shortcut that can be utilized to turn the plugin on and off.

Voice In is offered as a freemium.
The user has to pay for an annual subscription to be able to add custom text replacements.
This full-featured version of the plugin is called Voice-In Plus (VIP).
We will focus on VIP.

On activation of the VIP version of the plugin, the settings GUI page for custom commands is displayed for the user to use to enter commands either one by one through a GUI or by adding multiple voice commands through the text area that is opened after clicking on the bulk add button {ref}`fig:newSentence`.
The first option involves placing the voice trigger in one text area and the text replacement in the second text area.
The voice trigger does not need a comma after it, and the text replacement can span multiple lines without adding any markup, except that internal double quotes must be replaced with single quotes.
Any capitalization in the voice trigger will be ignored and written in lowercase.
The second option involves pasting in one or more lines of pairs of voice triggers and text replacements separated by commas, as in a CSV file.
In this option, text replacements that span more than one line must be enclosed with double quotes.
The internal double quotes must be replaced with single quotes; otherwise, the text replacement will be truncated at the position of the first internal double quote.

:::{figure} ./images/VoiceInNewSentence.png
:label: fig:newSentence
:width: 50%
Entering a single voice trigger and the corresponding command in Voice In Plus.
:::

The carrying capacity for the storage of voice-triggered commands is still being determined.
At one point, we had over 19,000 pairs of voice triggers and the corresponding text replacements.
Scrolling through the list of these voice commands was painful because it was too long.

This problem was efficiently resolved by exporting the commands to a CSV file, a process that significantly streamlined the management of voice-triggered commands.
Then, the stored commands on the website were cleared.
The exported CSV file was opened in a text editor, and the unneeded commands were selected and deleted.
In this case, that left about 7,000 commands.
The web page with the command library displayed could then be easily scrolled.
The practical size limit of the library is between 7,000 and 19,000 commands. This is because exceeding this limit may lead to performance issues, such as slower response times.

Amongst the first customizations are those supporting the generation of nonfiction writing.
Users may want to install our library of English expansions to avoid the tedium of converting English contractions to their expansions.
To avoid having to say `period` and `new line` at the end of each sentence when writing one sentence per line, the user can develop a custom command called `new sentence`, a combination of these two built-in commands. This command is useful when you want to dictate your text one sentence at a time.
Likewise, the custom command `new paragraph` can include a `period` followed by two `new line` commands, which works well when writing with blank lines between paragraphs and without indentation at the beginning of each paragraph.
Of course, these phrases will no longer be available in dictation because they will be used to trigger text replacements.

The VIP (Voice-Triggered Interactive Platform) documentation is integrated into the GUI (Graphical User Interface) that the user uses to configure VIP and carry out various tasks. The GUI is a visual interface that allows users to interact with the VIP system.

A dictation session with VIP is initiated by activating the plugin by clicking its icon.

There is a configuration page associated with the icon through which one can select the language and even a language's dialect.

VIP has several dozen built-in commands, some of which can be used to navigate the web page.

The voice-triggered snippets can be exported in a CSV file.
The file has two columns separated by a comma: the voice trigger and its text replacement. Each row represents a single voice-triggered command. The voice trigger is the phrase you say to activate the command, and the text replacement is the text that will be inserted when the command is triggered.
This file lacks a line of headers.

Multiple custom commands can be uploaded from a CSV file to the bulk add window of the plugin's configuration GUI. To create a custom command, you need to define the voice trigger and its corresponding text replacement in the CSV file, and then upload the file to the GUI.
Frequently, it is necessary to insert a code fragment that spans multiple lines.
This code fragment needs to be enclosed with double quotation marks.
It is not possible to use a backspace to escape internal pre-existing double quotation marks.
These pre-existing double quotes have to be replaced with single quotes.


## Discussion

The following discussion points, crucial for understanding the implementation of the ASR libraries we've described, have emerged.
We limit the discussion to the software that we have presented above.

### Independence from breaking changes in Jupyter

The Jupyter project lacks built-in support for code snippet libraries.
Due to the inherent limitations of the Jupyter project, the development of third-party extensions has become a necessity to support code snippets.
Unfortunately, changes in the core of Jupyter often break these extensions.
Users have to create Python environments for older versions of Jupyter to work with the snippets extension while missing out on the new features of Jupyter.
An obvious solution to this problem would be for the Jupyter developers to incorporate one of the snippet extensions into the base distribution of Jupyter to ensure that at least one form of support for snippets is always available.
Using voice-triggered snippets external to Jupyter side steps difficulty with breaking changes to Jupyter.

### Filling gap in tab-triggered snippets with voice-triggered snippets

Voice-triggered snippets, a promising innovation, offer a potential solution to the absence of extensions for Jupyter that support tab-triggered snippets.
Tab-triggered code snippets are standard in most text editors, whereas voice-triggered snippets have yet to become widespread in standard text editors.
One advantage of Jupyter Notebooks is that they run in the browser, where several automated Speech Recognition software packages operate (e.g., Voice-In Plus, Serenade, and Talon Voice).
We developed our libraries for Voice In Plus software because of its gentle learning curve and straightforward customization.
We did this to meet the needs of the broadest population of users.

### The role of AI-assisted voice computing

The dream of AI-assisted voice computing is to have one's intentions rather than one's words inserted into the document one is developing.
Our exposure to what is available through ChatGPT left us with an unfavorable impression due to the high error rate.
GitHub's copilot can also be used in LaTeX to autocomplete sentences.
Here again, many of the suggested completions need to be more accurate and require editing.
These autocompleted sentences slow down the user by getting in the way and leaving no net gain in productivity.

In addition, AI assistance in scientific writing has to be disclosed upon manuscript submission.
Some publishers will not accept articles written with the help of AI-writing assistants.
This could limit the options available for manuscript submission if one uses such an assistant and has the manuscripts rejected by a publisher that accepts such assistants.

### ASR extensions for Jupyter lab

We found three extensions developed for Jupyter Lab that enable speech recognition in Jupyter notebooks.
The first, [jupyterlab-voice-control](https://github.com/krassowski/jupyterlab-voice-control), supports custom commands and relies on the browser's language model.
This extension is experimental and not maintained; it does not work with Jupyter 4.2.
The second extension, [jupyter-voice-comments](https://github.com/Banpan-Jupyter-Extensions/jupyter-voice-comments),  relies on the DaVinci large language model to make comments in Markdown cells and request code fragments.
This program requires clicking on a microphone icon repeatedly, which makes the user vulnerable to repetitive stress injuries.
The third extension is [jupyter-voicepilot](https://github.com/JovanVeljanoski/jupyter-voicepilot).
Although the extension's name suggests it uses GitHub's Copilot, it uses whisper-1 and ChatGPT3.
This extension requires an API key for ChatGP3.
The robustness of our approach is that the Voice-In Plus software will always operate within Jupyter Lab when Jupyter is run on a web server.


### Caveats about voice computing
We found five caveats to doing voice computing.
These points reflect the imperfect state of available language models.
We suggest how to cope with these limitations while improving productivity.

First, the rate at which one speaks is a crucial variable.
If you speak too slowly the words in a voice trigger that is a compound word, your words may not be interpreted as the intended voice trigger.
Instead, the individual words will be printed on the screen.
On the other hand, if one speaks too quickly, one may get ahead of the language model, which may stall.
If the plugin is not responding, it is best to restart your connection with the language model by inactivating and restarting the plugin.
I can generally dictate three to seven paragraphs before the software falls behind and halts.

Second,  the language model may have difficulty with specific words or phrases.
This is a common experience, which is rectified by using text replacements.
A difficult-to-interpret word or phrase may cause the language model to return a series of alternate words or phrases that were not intended.
The solution to this problem is to map these alternate phrases to the desired phrase to ensure it is returned correctly.
Invariably, some of one's mappings may get invoked when not intended.
This event is rare enough to be tolerated.
The large language models are imperfect, and these difficulties are still widespread.
It is expected that over the next several years the language models will continue to evolve and improve, making these difficulties less common.
Nonetheless, the ability to map the alternate phrases to the desired phrase demonstrates the great value of using text replacements to achieve the desired outcome.

Third, language models vary quite a bit in terms of their requirements for an excellent microphone.
Newer language models can often accurately transcribe your words using the internal microphone from your laptop or desktop computer.
Contrary to the prevailing advice in some quarters, a high-quality external microphone may not be required.
The microphone in our 2018 MacBook Pro works well with Voice In Plus.

Fourth, one can inadvertantly change the case of words while dictating in Voice In Plus.
To switch back to the default case, one need to navigate to the options page and select the text transform button to open a GUI that lets you set the case globally.
This event occurs about once every 100 hours of dictation.

Fifth, a related problem is the inadvertent activation of other voice computing software on one's computer.
For example, once in about 100 hours of dictation, one will say a phrase that resembles `Hey, Siri`.
*Siri* will then respond.
One solution is to inactivate *Siri* so that it cannot respond to one's speach.

These caveats are minor annoyances.
We think that productivity gains out of the disruptions caused by these annoyances.


### Common hazards when voice computing

In our year of using voice control daily, we have encountered two mishaps.
First, we have accidentally recorded a conversation when someone walked into my office while doing computing.
If we fail to turn off the ASR software, bits of our conversation are recorded at the mouse cursor's position.
This inserted text has to be deleted later.
This is a bigger problem when editing a code file or code cell.
The injection of unwanted words can introduce bugs that take time to remove.

Second, some ASR software may become activated upon restarting the computer.
If their state is overlooked, words from one's speech, a YouTube video, or a Zoom meeting can be converted into computer commands that get executed in unintended ways.
It can be embarrassing if it occurs in the middle of a Zoom meeting.
Also, two voice-control software can be activated simultaneously, and the speech will be transcribed twice in the text area.


### Future directions

One future direction is to build out the libraries of voice-triggered snippets that have been developed.
Another direction includes developing a method for facilitating voice stops analogous to tab stops in code snippets for advanced text editors.
These voice stops would advance the cursor to all sites that should be considered for editing to customize the code snippet for the problem.
The other related advance would be mirroring the parameter values at identical voice stops.
