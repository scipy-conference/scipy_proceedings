---
# Voice Computing with Python in Jupyter Notebooks
title: Voice Computing with Python in Jupyter Notebooks
abstract: |
    Jupyter is a popular platform for writing literate programming documents that contain computer code and its output interleaved with prose that describes the code and the output. It is possible to use one's voice to interact with Jupyter notebooks. This capability opens up access to those with impaired use of their hands. Voice computing also increases the productivity of workers who are tired of typing, and increases the productivity of those workers who speak faster than they can type. Voice computing can be divided into three activities: speech-to-text, speech-to-command, and speech-to-code. Several automated speech recognition software packages operate on Jupyter notebooks and support these three activities. We will provide examples of these activities as they pertain to applications of Python in our research on the molecular structures of proteins and nucleic acids important in medicine. Several software tools at MooersLab on GitHub facilitate the use of voice computing software in Jupyter.
---


## Introduction


Jupyter notebooks provide a highly interactive computing environment where users run Markdown and code cells to yield almost instant results.
This form of interactive computing provides the instant gratification of seeing the results of the cells' execution; this might be why Jupyter is so popular for data analysis [@Perkel2021TenComputerCodesThatTransformedScience].
The most popular modality for interacting with the Jupyter notebooks is to use the keyboard and the computer mouse.
However, there are opportunities to generate prose and code using one's voice instead of one's hands.
While those who have lost use of their hands must rely solely on their voice, other users can enhance their prose generation with their voice to boost their productivity and give their hands a rest when fatigued from typing. This inclusive approach enhances the Jupyter notebook experience for all users, regardless of their physical abilities.
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

The simplest text replacements involved the replacement of English contractions with their expansions {ref}`fig:contractions`.
English contractions are not used in formal writing for many reasons.
Many of the automatic speech recognition software packages will default to using contractions because the audience for the software are people who are writing informally for social media where English contractions are acceptable.
By adding the library that maps contractions to their expansions, the expansion will be inserted whenever the contraction is used otherwise.
This automated replacement of English contractions saves time during the editing process.

:::{figure} ./images/VoiceInCustomCommandsContractions.png
:alt: contractions
:class: bg-primary
:width: 60%
:align: center
:label: fig:contractions

Webpage of custom commands. The buttons are used to edit existing commands and add new commands. Three English contractions and their expansions are shown.
:::

The contents of the library can be further organized by the first word of the voice trigger {ref}`table:commands`.
This first word is often a verb.
For example, the word `expand' is used to expand acronyms.
Acronyms must be defined upon their first use.

One's memory is often inaccurate about the words represented by the letters in the acronym.
The library ensures that the correct expansion of the acronym is used.
This can save time that would otherwise be spent looking up expansions.
We have included acronyms widely used in Python programming, scientific computing, data science, statistics, machine learning, and Bayesian data analysis.
These commands are found in domain-specific libraries, so a user can select which voice triggers they would like to add to their private collection of voice commands.

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
:class: bg-primary
:width: 100%
:align: center
:label: fig:displayeq

Three snapshots from a Zoom video of using the voice-trigger *display electron density equation* in a Markdown cell in a Jupyter notebook. A. The Zoom transcript showing the spoken voice trigger. B. The text replacement in the form of a math equation written in LaTeX in display mode in the Markdown cell. C. The rendered Markdown cell. The green and black tab on the right of each panel indicates that the Voice In plugin is active and listening for speech.
:::

Another example of a verb starting a voice trigger is the command `display  <equation name>`.
This command is used in Markdown cells to insert equations in the display mode of LaTeX in Markdown cells.
For example, the voice trigger `display the electron density equation` is shown in the transcript of a Zoom video {ref}`fig:displayeq`. .
This is followed by the image in the middle that shows the text replacement in the form of a LaTeX equation in the display mode.
This image is followed by the resulting Markdown cell after it is rendered by running the cell.

Likewise, the command `inline <equation name>` is used to insert equations in sections of prose in Markdown cells.
We have made available voice-triggered snippet libraries of equations commonly found in Bayesian data analysis and structural biology.
These libraries are incomplete, but they can provide a source of inspiration.
They can also be used as templates for making new libraries for one's particular domain.

Some voice triggers start with a noun.
For example, the voice trigger `URL` is used to insert URLs for important websites.
Another example involves the use of the noun `list` in the voice trigger  as in `list matplotlib color codes` to generate the list of the color codes used in Matplotlib plots.
This kind of information is useful when developing plots with Matplotlib because it can save you the trouble looking up this information: The software documentation will literally be at the tip of your tongue.

The markup language code is inserted by using the verb *insert* followed by the markup language name and the name of the code.
For example, the command `insert markdown itemized list` will insert five vertically aligned dashes to start an itemized list.
The command `insert latex itemized list` will insert the corresponding code for an itemized list in LaTeX.

We have developed a library specifically for the flavor of [Markdown](https://github.com/MooersLab/markdown-jupyter-voice-in/blob/main/markdown.csv) utilized in Jupyter notebooks.
This library is used to insert the appropriate Markdown code in Markdown cells.

We have included a [library for LaTeX](https://github.com/MooersLab/latex-voice-in) because tex files can be edited by Jupyter Lab.
According to the above rule, we should use `insert latex equation` to insert the code for the equation environment.
However, we broke this convention by omitting that word latex to make the commands more convenient because this is our default, typesetting language that we use every day.
We were already comfortable with using the abbreviated commands.

We have not figured out how to use voice commands to advance the cursor in a single step to sites where edits should be made in analogy to tab stops in conventional snippets.
Instead, the built-in voice commands can be utilized to move the cursor forward or backwards and for the purpose of selecting replacing words.
We included the markup associated with the yasnippet snippet libraries to serve as a benchmark for users to recognize the sites that should be considered for modification to customize the snippet for their purpose.

### Libraries for code cells

The libraries for code-cells utilize the `insert` command to insert chunks of Python code.
We try to avoid making voice commands for small fragments of code that might fit on a single line.
An exception to this was the inclusion of a collection of one liners that are in the form of several kinds of comprehensions.
As mentioned above, we have developed chunks of code for the purpose of performing specific functions.
These chunks of code could be functions, or they just could be lines of code that produce an output in the form of a table or plot or some form of analysis.
The idea was to be able to fill a code cell with all the code required to produce the desired output.
Unfortunately, the user will still have to use their voice or computer mouse to move the cursor back over the code chunk and customize portions of the code chunk as required for the task at hand.

While self-contained examples that utilize fake data are useful for the purpose of illustrating concepts, these examples are very frustrating for beginners who need to read in their own data and who would like to apply the code example to their problem at hand.
The reading in appropriately cleaned data is a common task in data science, and it is also a common barrier applying Jupyter notebooks to Scientific problems.
We provide code fragments in a data wrangling library that support the importing of several file types directly for downstream utilization as well as for the purpose of importing the data into numpy, pandas, and several of the more recently developed programs that is support multidimensional data structures.

After the inputting of data, it needs to be displayed in a tabular format for inspection that to check that it was properly imported and also to carry out basic summarization statistics by column and row.

After the data have been verified as being properly imported, there has often the need to explore that data by plotting it to detect relationships between the parameters of a model and the output.
We strove to focus on the matplotlib library for the purpose of generating a wide diversity of plots [@matplotlib].
The code fragments that we develop cover the most commonly used plots such as Scatter Plots, bar graphs (including horizontal bar graphs), kernel density fitted distributions, heat Maps, pie charts, contour plots, and so on.
We also include examples of 3D plots.
We include a variety of variance in terms of the for batting of the tick marks and access labels as well as the keys and the form of the wines so bad users can use this information as templates to generate plots for their own purpose.
The generation of plots with lines of different shape in terms of whether they are solid or have dashes are dotted or have combinations thereof is essential because plots generated with just color blinds are vulnerable to having that their information compromised when printed in black and white.
The latter is often done at institutions that are trying to cut costs.
Although we provide some examples from some of the higher-order plotting programs like Seaborn, we focused on matplotlib because most of the other plotting programs are built on top of it, with the exception of the interactive plotting programs.

We also support the import of external images.
This is often overlooked, but these externally derived images are often important parts of the story that is being told by the Jupyter notebook.

### Jupyter specific library

We provide a [library](https://github.com/MooersLab/jupyter-voice-in/blob/main/jupyter.csv) of 85 cell and line magics  that facilitate the Jupyter notebook's interaction with the rest of the operating system.
The cell magics have the prefix *cell magic*, and the line magics have the preflx *line magic*.
For example, the voice command *line majic run* insert `%run`; it is used to run script files after supplying the name of the script file as also shown in {fig}`quiz`.



### Interactive quiz

We developed a [quiz](https://github.com/MooersLab/voice-in-basics-quiz) to improve recall of the voice commands .
These quizzes are interactive and can be run in the terminal or in Jupyter notebooks {ref}`fig:quiz`.
The latter can be saved to keep a record of one's performance on a quiz.


:::{figure} ./images/runningQuiz.png
:label: fig:quiz
:width: 130% 
An example of an interactive session with a quiz in a Jupyter notebook. The code for running the quiz was inserted into the code cell with the voice command `run voice in quiz`.
:::

To build long-term recall of the commands, one must take the quiz five or more times on alternate days according to the principles of spaced repetition learning.
These principles were developed by the German psychologist Hermann Ebbinghaus in the last part of the 19th Century.
They have been validated several times by other researchers.
Space repetition learning is one of the most firmly established results of research into human psychology.

Most people lack the discipline to carry out this kind of learning because they have to schedule the time to do the follow-up sessions.
Instead, we anticipate that most people will take these quizzes several times in a half hour before they spend many hours utilizing the commands.
If that use occurs on subsequent days, then recall of the alphabet will be reinforced, and it may not be necessary to take the quiz again.


### Voice In Plus

Voice In is a plug-in for Google Chrome and Microsoft Edge web browsers that uses the Google API.
The plug-in operates in most text areas of web pages.
These text areas include those of web-based email packages and online sites that support distraction-free writing.
These text areas also include the Markdown and code cells of Jupyter notebooks.
Voice In also works in various plain text documents opened in Jupyter Lab for online writing.
Obviously, Voice In will not work in standalone applications that support the editing of Jupyter notebooks, such as the Jupyter Lab app, the nteract app, and external text editors, such as VS Code, that support the editing of Jupyter notebooks.

After Voice-In Plus is activated, it will listen for words for 3 minutes before automatically shutting down.
It is very accurate with a word error rate that is well below 10\%.
It can pick out words in spite of background ambient noise, such as traffic or bird songs.
The language model is quite robust in that dictation can be performed out without the use of an external microphone.
For example, the built-in microphone available in the MacBook Pro laptop computer is sufficient.
In contrast, other VSR software requires high-quality external microphones.
The need to use an external microphone imposes a motivational barrier.

Because of the way the system is set up to utilize the Google API, there is not much of a latency issue.
The spoken words' transcriptions occur nearly in real-time; there is only a minor lag.
The program can generally keep up with dictation occurring at a moderate pace for at least several paragraphs, whereas competing systems tend to quit after one paragraph.
The program tends to hallucinate only when the dictation has occurred at high speed because the transcribing falls behind.
As a result, the user has to pay attention to the progress of the transcription.
If the transcription halts, it is best to deactivate the plugin, activate it, and resume the dictation.
Great care has to be taken to pronounce the first word of the sentence loudly such that it will be recorded; otherwise, this first word will likely not be recorded.
This problem is most acute when there has been a pause in the dictation.

The software does not automatically insert punctuation marks.
The user has to vocalize the name of the punctuation mark that they want inserted.
These also have to utilize a built-in new-line command to start new lines.
The user has to develop the habit of using this command if they write one sentence per line.
This latter form of writing is very useful for first drafts because it greatly eases the shuffling of the order of sentences during rewriting.
This form of writing is also very compatible with Virgin control systems like git because the changes can be tracked more easily by line number.

The program can sometimes be unresponsive.
In these situations, the plug-in can be turned on and then again.
This act will restore normal behavior.

Voice In has an associated configuration GUI where the user can customize a number of settings.
The first setting to be set is the language that will be used during dictation.
There is support for a number of foreign languages and for different dialects of English.
The user can also configure a keyboard shortcut that can be utilized to turn the plug-in on and off.
I use the option-L key combination to turn the plug-in on and off.

Voice In is offered as a freemium.
The user has to pay for an annual subscription to be able to add custom text replacements.
This full-featured version of the plugin is called Voice-In Plus (VIP).
This is the version upon which we will focus.

On activation of the VIP version of the plug-in, the settings GUI page for custom commands is displayed for the user to use to enter commands either one by one through a GUI or by adding multiple voice commands though the text area that is opened after clicking on the bulk add button {ref}`fig:newSentence`.
The first option involves placing the voice trigger in one text area and the text replacement in the second text area.
The voice trigger does not need a comma after it, and the text replacement can span multiple lines without adding any markup, except that internal double quotes must be replaced with single quotes.
Any capitalization in the voice trigger will be ignored and written in lowercase.
The second option involves pasting in one or more lines of pairs of voice triggers and text replacements separated by commas, as you would in a CSV file.
In this option, text replacements that span more than one line must be enclosed with double quotes.
Obviously, the internal double quotes have to be replaced with single groups; otherwise the tex replacement will be truncated at the position of the first internal quote.

:::{figure} ./images/VoiceInNewSentence.png
:label: fig:newSentence
:width: 50%
Entering a single voice trigger and the corresponding command in Voice In Plus.
:::


The carrying capacity for the storage of voice-triggered commands is unclear.
At one point, I had over 19,000 pairs of voice triggers and the corresponding text replacements.
Scrolling through the list of these voice commands was painful because it was too long.

This problem was remedied by exporting the commands to a CSV file.
Then the stored commands on the website were cleared.
The exported CSV file was opened in a text editor and the unneeded commands were selected and deleted.
In this case, that left about 7,000 commands.
The web page with the command library displayed could then be easily scrolled.
The practical size limit of the library is between 7,000 and 19,000 commands.

Amongst the first customizations to be made are those supporting the generation of nonfiction writing.
Users may want to install our library of English expansions to avoid the tedium of converting English contractions to their expansions.
To avoid having to say `period` and `new line` at the end of each sentence when writing one sentence per line, the user can develop a custom command called `new sentence`, which is a combination of these two built-in commands.
Likewise, the custom command `new paragraph` can include a `period` followed by two `new line` commands, which works well when writing with blank lines between paragraphs and without indentation at the beginning of each paragraph.
Of course, these phrases will no longer be available for use in dictation because they will be used to trigger text replacements.

The VIP documentation is integrated into the GUI that the user uses to configure VIP and carry out various tasks.

A dictation session with VIP is initiated by activating the plugin by clicking on its icon.

There is a configuration page associated with the icon through which one can select the language are and even dialect of a language.

VIP has several dozen built-in commands, some of which can be used to navigate the web page.

The voice-triggered snippets can be exported in a CSV file.
The file has two columns separated by a comma: the voice-trigger and its text replacement.
This file lacks a line of headers.

Multiple custom commands can be uploaded from a CSV file to the *bulk add* window of the plugin's configuration GUI.
Frequently, it is necessary to insert a code fragment that spans multiple lines.
This code fragment needs to be enclosed with double quotation marks.
It is not possible to use a backspace to escape internal pre-existing double quotation marks.
These pre-existing double quotes have to be replaced with single quotes.

## Discussion

The following discussion points arose during our implementation of the ASR libraries described above.
We limit the discussion to the software that we have presented above.

### Independence from breaking changes in Jupyter

The Jupyter project lacks built-in support for code snippet libraries.
Instead, third parties have developed several extensions for Jupyter to support code snippets.
Unfortunately, changes that occur in the core of Jupyter often break these extensions.
Users have to create Python environments for older versions of Jupyter work with the snippets extension while missing out on the new featuers of Jupyter.
An obvious solution to this problem would be for the developers of Jupyter to incorporate one of the snippet extensions into the base distribution of Jupyter to ensure that at least one form of support for snippets is always available.
The use of voice-triggered snippets external  to Jupyter side steps difficulties with breaking changes to Jupyter.

### Filling gap in tab-triggered snippets with voice-triggered snippets

Voice-triggered snippets also provided an opportunity to overcome the absence of extensions for Jupyter that support tab-triggered snippets.
Tab-triggered code snippets are standard in most text editors, whereas voice-triggered snippets have yet to become widespread in standard text editors.
One advantage of Jupyter Notebooks is that they run in the browser, where several automated Speech Recognition software packages operate (e.g., Voice-In Plus, Serenade, and Talon Voice).
We developed our libraries for Voice In Plus software because of its gentle learning curve and simple customization.
We did this to meet the needs of the widest population of users.

### The role of AI-assisted voice computing

The dream of AI-assisted voice computing is to have one's intentions rather than one's words inserted into the document you are developing.
Our exposure to what is available through ChatGPT left us with an unfavorable impression due to the high error rate.
GitHub's copilot can also be used in LaTeX to autocomplete sentences.
Here again, many of the suggested completions are inaccurate and require editing.
These autocompleted sentences tend to slow down the user by getting in the way and leaving no net gain in productivity.

In addition, the utilization of AI assistance in scientific writing has to be disclosed upon manuscript submission.
Some publishers will not accept articles written with the help of AI-writing assistants.
This could limit the options available for manuscript submission should one use such an assistant and have the manuscripts rejected by a publisher that accepts such assistants.

### ASR extensions for Jupyter lab

We found three extensions developed for Jupyter Lab that enable the use of speech recognition in Jupyter notebooks.
The first, [jupyterlab-voice-control](https://github.com/krassowski/jupyterlab-voice-control) supports the use of custom commands and relies on the language model in the browser.
This extension is experimental and not maintained; it does not work with Jupyter 4.2.
The second extension, [jupyter-voice-comments](https://github.com/Banpan-Jupyter-Extensions/jupyter-voice-comments),  relies on the DaVinci large language model to make comments in Markdown cells and request code fragments.
This program requires clicking on a microphone icon repeatedly, which makes the user vulnerable to repetitive stress injuries.
The third extension is [jupyter-voicepilot](https://github.com/JovanVeljanoski/jupyter-voicepilot).
Although the name of the extension suggests it uses GitHub's Copilot, it uses whisper-1 and ChatGPT3.
This extension requires an API key for ChatGP3.
The robustness of our approach is that the Voice-In Plus software will always operate within Jupyter Lab when Jupyter is run in a web.


### Caveats about voice computing
We found five caveats to doing voice computing.
These points reflect the imperfect state of available language models.
We suggest how to cope with these limitations while improving productivity.

First, the rate at which you speak is an important variable.
If you speak too slowly a voice trigger that is a compound word, your words may not be interpreted as the intended voice trigger.
Instead, the individual words will be printed to the screen.
On the other hand, if you speak too quickly, you may get ahead of the language model and it may stall.
If the plugin is not responding, it is best to restart your connection with the language model by inactivating the plugin and restarting it.
I can generally dictate three to seven paragraphs before the software falls behind and halts.

Second,  the language model may have a difficult time with a specific words or phrases.
This is a common experience, which is rectified by using text replacements.
A difficult-to-interpret word or phrase may cause the language model to return a series of alternate words or phrases that were not intended.
The solution to this problem is to map these alternate phrases to the desired phrase to ensure that it is returned correctly.
Invariably, some of your mappings may get invoked when not intended.
This event is rare enough to be tolerated.
The large language models are not perfect, and these sorts of difficulties are still widespread.
It is expected that over the next several years the language models will improve further and that these difficulties will become less common.
Nonetheless, the ability to map the alternate phrases to the desired phrase demonstrates the great value of being able to use text replacements to get the desired outcome.

Third, language models vary quite a bit in terms of their requirements for an excellent microphone.
Newer language models often can accurately transcribe your words using the internal microphone that comes with your laptop or desktop computer.
A high quality external microphone may not be required, contrary to the prevailing advice in some quarters.
The microphone in our 2018 MacBook Pro works well with Voice In Plus.

Fouth, you can inadvertantly change the case of words while dictating in Voice In Plus.
To switch back to the default case, you need to navigate to the options page and select the text transform button to open a GUI that lets you set the case globally.
This event occurs about one every 100 hours of dictation.

Fifth, a related problem is the inadvertent activation of other voice computing software on your computer.
For example, once in about 100 hours of dictation, I will say a phrase that resembles `Hey, Siri`.
*Siri* will then respond.
One solution is to inactivate *Siri* so that it cannot respond to your speach.

These caveats are minor annoyances.
We think that the productivity gains out wiegh the disruptions caused by these annoyances.


### Common hazards when voice computing

In my year of using voice control every day, I have encountered two kinds of  mishaps.
First, I have accidentally recorded a conversation when someone walked into my office while doing computing.
If I fail to turn off the ASR software, bits of our conversation are recorded at the positon of the mouse cursor.
This inserted text has to be deleted later.
This is a bigger problem when a code file or code cell is being edited.
The injection of unwanted words can introduce bugs that take time to remove.

Second, some ASR software may become activated upon restarting the computer.
If their state is overlooked, words from your speeh, a YouTube video, or a Zoom meeting can be converted into computer commands that get executed in unintended ways.
If this occurs in the middle of a Zoom meeting, this can be embarrassing.
Also, two voice-control software can activated at the same time and speach can be transcribed twice in the text area.


### Future directions

One future direction is to build out the libraries of voice-triggered snippets that have been developed to date.
Another direction includes the development of a method of facilitating voice stops, in analogy to tab stops in code snippets for advanced text editors.
These voice stops would advance the cursor to alll sites that should be considered for edting to customize the code snippet for the problem at hand.
The other related advance would be the mirroring of the parameter values at identical voice stops.
