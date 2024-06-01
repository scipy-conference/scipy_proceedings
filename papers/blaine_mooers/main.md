---
# Voice Computing with Python in Jupyter Notebooks
title: Voice Computing with Python in Jupyter Notebooks
abstract: |
    Jupyter Notebook is a popular platform for writing literate programming documents that contain computer code and its output interleaved with prose that describes the code and the output. It is possible to use one's voice to interact with Jupyter notebooks. This capability opens up access to those with impaired use of their hands. Voice computing also increases the productivity of workers who are tired of typing, and increases the productivity of those workers who speak faster than they can type. Voice computing can be divided into three activities: speech-to-text, speech-to-command, and speech-to-code. Several automated speech recognition software packages operate on Jupyter notebooks and support these three activities. We will provide examples of all three activities as they pertain to applications of Python to our research on the molecular structures of proteins and nucleic acids important in medicine. Several software tools at MooersLab on GitHub facilitate the use of voice computing software in Jupyter.
---


## Introduction

Jupyter notebooks provide a highly interactive computing environment where Markdown and code cells are run to yield almost instant results [@Perez2015ProjectJupyterComputationalNarrativesAsTheEngineOfCollaborativeDataScience].
This form of interactive computing provides the instant gratification of seeing the results of the execution of the cells [@Perkel2021TenComputerCodesThatTransformedScience].
The most popular modality for interacting with the Jupyter notebook is to use the keyboard and the computer mouse.
However, there are opportunities to generate prose and code using one's voice instead of one's hands.
While those who have lost use of their hands are forced to use only their voice, other users can augment their typing with their voice to enhance their productivity and perhaps give their hands a break when tired of typing.
In other words, most users can use their voice to complement the use of the keyboard.
For example, dictation of prose in Markdown cells is an obvious application of voice computing in Jupyter.
The ease of generating prose via speech can help promote more complete descriptions of computations executed in adjacent blocks by lowering the barrier to generating prose.

Some Speech-to-text software also supports the mapping of a word or phrase to a text replacement; there are many ways of exploiting text replacements in Markdown and code cells.
For Markdown cells, we have mapped the English contractions to their expansions, so whenever we say a contraction, the expansion is automatically inserted.
This reduces the amount of editing downstream. 
Another class of text replacements is the expansion of acronyms into the phrase they represent.
The BibTeX cite keys for common references can mapped to a command like `cite key for scipy`.
Equations type set in LaTeX for rendering with MathJaX can also be mapped to commands like `inline pythargeous theorem` and `display electron density equation`, depending on whether the equation is to be *in-line* in a sentence or centered and  isolated in *display-mode*.
Voice commands can be mapped to tables, templates, and even software licenses to be inserted as text replacements. 
For Jupyter code cells, voice commands can be mapped to chunks of code of various sizes.
We call these voice commands that trigger a text replacement *voice triggers* in analogy to tab triggers with conventional tab-triggered snippets in advanced text editors.

To ease the use of voice commands in Jupyter notebook cells, we have developed sets of voice-triggered snippets for use in either markdown or code cells.
We are building our prior experience with tab-triggered code snippets in text editors [@Mooers2021TemplatesForWritingPyMOLScripts] and domain specific code snippet libraries for Jupyter [@Mooers2021APyMOLSnippetLibraryForJupyterToBoostResearcherProductivity].
We have made several libraries of these voice-triggered snippets for several of the popular modules of the scientific computing stack for Python.
Although the Jupyter environment supports polyglot programming, we have restricted our focus to Python and Markdown.
While some code snippets are one-liners, most code snippets span many lines and perform a complete task, such as generating a plot from a data file.
Plot generation is a common data science task that is hindered by difficulties with importing data from external files and by finding the parameters and settings required to generate the desired plot.
We tried to make these code snippets complete in terms of their ability to perform a function, such as generating standard kinds of plots with Matplotlib.
The snippets are supplied with example data to ease the adaptation of these plotting codes by beginners.
These libraries provide code that is known to work, unlike the situation with chatbots, which do not always return working code.
The high reliability of code snippet libraries suggests that the demand for them will presist through the current AI hypercycle.

## Methods and Materials

### Construction of the snippet libraries
Some voice snippets had already been used for a year for the purpose of composing prose by using dictation.
These snippets were gathered together in modular files to ease their selective use.
The voice snippets were translated into formats appropriate for the automated speech recognition (ASR) software Voice-In Plus while retaining the voice trigger where possible.
Universal voice triggers ease moving from one ASR package to another.

The voice snippets were stored in CSV files for the Voice In Plus software.
The contents of these files can be copied and pasted into the bulk add text area through the Voice In Plus configuration GUI.

### Construction of interactive quizzes
To aid the mastery of the ASR syntax, we developed interactive quizzes.
The quizzes were written in  Python and can be run in the terminal are in Jupyter notebooks.
The questions, answers, and exclamations were stored in a tuple.
The questions in a quiz were stored in a list of these tuples.
The order of the questions was randomized upon restart of the quiz.
The inspiration for the quiz came from an example in the book *Automate the boring stuff with Python* [@Sweigart2015AutomateTheBoringStuffWithPythonPracticalProgrammingForTotalBeginners].

The wrongly answered questions were fed back to the user after they failed to correctly answer five questions.
This recycling of the wrongly answered questions helps to build up the recall of the correct answers.
The number of questions in a quiz is limited to 40 for the purpose of not exhausting the user.

A function writes out the quiz to a PDF file upon completion of the quiz or upon early exiting the quiz.
This PDF enables the printing of a quiz so that the paper version can taken while away from the computer.

### Availability of the libraries and quizzes
The libraries were tested utilizing Jupyter Lab version 4.2 and Python 3.12 installed from Macports.
All libraries are made available on GitHub for direct download.


## Results

First, we describe the contents of the snippet libraries and the kinds of problem they can solve.
We attempt to group the libraries into several categories to simplify their explanation.
Second, we describe the deployment of the snippet libraries for the automated speech recognition (ASR) software package Voice In Plus.
The results section that follows describes the libraries in the same order.

The Voice In Plus software runs only in the Web browser, but is the easiest to customize and has the gentlest learning curve.
The first package requires an Internet connection to run, and the second package can be switched from using a local language model to one of three servers.
The user can choose the desired server.

### Composition of the libraries

We describe the libraries that we have developed to facilitate our own workflows.
Our descriptions of these libraries are meant to illustrate how voice-triggered snippets are used with automated speech recognition software.
The developers in our audience will either want to augment the libraries that we have developed with their own commands or they will want to use our libraries as templates to develop their own libraries independently.

The libraries are made available in a modular format so that the user can select sets of commands that are most useful for their workflow.
In general, our libraries are quite broad in nature, so they meet the needs of most users.
There are several libraries that are domain-specific.
These libraries can inspire the development of libraries for other domains.
The contents of the libraries can be divided into two categories.
One subset of libraries supports carrying out dictation about science in the markdown cells of Jupyter Notebooks while the other subset supports writing scientific Python in code cells.
While some code is specific to Jupyter, such as line and cell magics, most of the voice-triggered snippets can be used in Markdown files and Python scripts that are being edited in Jupyter Lab.

### Libraries for markdown cells

These libraries consist of a short phrase that is replaced with another phrase or with computer code.
The short phrase to be replaced is called the voice-trigger, in analogy to the tab-trigger in text editors.
Tab-triggers are typed with a keyboard and then the tab key is entered to trigger the insertion of the corresponding snippet of code.
Often the tab trigger name is under tab completion, so in some text editors the tab key will have to be hit twice: the first tab is used to auto-complete the tab trigger name.
The second tab then triggers the insertion of the corresponding code.
This two-step process has the advantage of allowing the user to select a different tab trigger before inserting the code.
Voice triggers differ in that there is no analog of a tab key required to insert the code.
There is no opportunity to revise the voice trigger, so the insertion at the wrong code will have to be undone with the undo command.

The simplest text replacements involved the replacement of English contractions with their expansions.
English contractions are not used in formal writing for many reasons.
Many of the automatic speech recognition software packages will default to using contractions because the audience for the software are people who are writing informally on for social media where English contractions are acceptable.
By adding the library that maps contractions to their expansions, the expansion will be inserted whenever the contraction is used otherwise.
This automated replacement of English contractions saves time during the editing process.

:::{figure} ./images/VoiceInCustomCommandsContractions.png
:label: fig:contractions
Webpage of custom commands. The buttons are used to edit existing commands and add new commands. Three English contractions and their expansions are shown.
:::

The contents of the library can be further organized by the first word of the voice trigger.
This first word is often a verb.
For example, the word `expand' is used to expand acronyms.
Acronyms must be defined upon their first use.
One's memory is often inaccurate about the words represented by the letters in the acronym.
The library ensures that the correct expansion of the acronym is used.
This can save a lot of time by looking up expansions.
We have included acronyms widely used in Python programming, scientific computing, data science, statistics, machine learning, and Bayesian data analysis.
These commands are found in domain-specific libraries, so a user can select which voice triggers they would like to add to their private collection of voice commands.

- **expand** acronyms
- **the book** title
- **email** dsw (i.e., inserts list of e-mail addresses)
- **insert** code blocks (including equations in LaTeX)
- **list** (e.g., font sizes in \LaTeX, steps in a protocol, members of a committee, etc.)
- **open** webpage (e.g., open pytexas, open google scholar)
- **display** equation in display mode (e.g., display electron density equation)
- **inline** equation in-line (e.g., inline information entropy equation)

Another example of a verb starting a voice trigger is the command `display  <equation name>`.
This command is used in markdown cells to insert equations in the display mode of LaTeX in Markdown cells.
For example, the voice trigger `display the electron density equation` is shown in the transcription of a Zoom video in figure 1A .
This is followed by the image in the middle that shows the tax replacement in the form of a LaTeX equation in the display mode.
This image is followed by the resulting markdown cell after it is rendered.
Likewise, the command `inline <equation name>` is used to insert equations in sections of prose in markdown cells.
We have made available voice-snippet libraries of equations commonly found in machine learning, statistics, Bayesian data analysis, physics, chemistry, biophysics, structural biology, and data science.
These libraries are incomplete, but they can provide a source of inspiration.
They can also be used as templates for making new libraries for one's particular domain.

Some voice triggers start with a noun.
For example, the voice trigger `URL' is used to insert URLs for important websites.
Another example involves the use of the noun `list` in the voice trigger  as in `list <matplotlib color codes>` to generate the list of the color codes used in Matplotlib plots.
This kind of information is useful when developing plots with Matplotlib because it can save you the trouble looking up this information: The software documentation will literally be at the tip of your tongue.
Although the documentation is quite accessible, it takes much longer to find information on the Web than is commonly admitted.
Likewise, each Web search for information goes down a path that is filled with attention-grabbing distractions that can easily led one to be sidetracked from their original task.

We have developed a library specifically for the flavor of markdown utilized in Jupyter notebooks.
This library can be utilized to use voice commands to insert the appropriate markdown code in markdown cells.
We have included a library for LaTeX because these documents can be composed in tex files that are being edited by Jupyter Lab.
We have also included a library for the markedly structured text markdown (MyST markdown) that is being developed to integrate the output from Jupyter notebooks into scientific publishing.

The markup language code is inserted by using the verb ``insert'' followed by the markup language name and the name of the code.
For example, the command ``insert markdown itemized list'' will insert five vertically aligned dashes to start an itemized list.
The command `insert latex itemized list` will insert the corresponding code for an itemized list in LaTeX.

While these libraries have voice-triggered snippets, they lack voice stops where one uses their voice to advance to sites that need to be edited in the code fragment.
These voice stops would be in analogy with the tab stops in conventional snippet libraries where one hits the tab after inserting the snippet for the purpose of advancing the cursor to sites that should be edited in order to adapt the code to the problem at hand.
Often, the tab stops are mirrored at sites that share the same parameter value.
Because of the mirroring, a change in the parameter value at one site will be propagated to the other sites.
The propagation of these changes insures that all the sites were changed; this avoids downstream debugging.

We have not figured out how to use voice commands to advance the cursor to sites where edits should be made
Voice commands can be utilized in some of the automated speech recognition software for the purpose of moving the cursor forward or backwards and for the purpose of selecting replacing words.
We have left the markup associated with the yasnippet snippet libraries in place to serve as a benchmark for users to recognize the sites that should be considered for modification to  customize the snippet for their purpose.

### Libraries for code cells

The libraries for code-cells utilize the ``insert'' command to insert chunks of Python code.
We try to avoid making voice commands for small fragments of code that might fit on a single line.
An exception to this was the inclusion of a collection of one liners that are in the form of several kinds of comprehensions.
As mentioned above, we have developed chunks of code for the purpose of performing specific functions.
These chunks of code could be functions, or they just could be lines of code that produce an output in the form of a table or plot or some form of analysis.
The idea was to be able to fill a code cell with all the code required to produce the desired output.
Unfortunately, the user will still have to use their voice or computer mouse to move the cursor back over the code chunk and customize portions of the code chunk as required for the task at hand.
Wherever possible, we included the boilerplate needed for documenting the code chunk in the function in the class Downstream.

While self-contained examples that utilize fake data are useful for the purpose of illustrating concepts, these examples are very frustrating for beginners who need to read in their own data and who would like to apply the code example to their problem at hand.
The reading in appropriately cleaned data is a common task in data science, and it is also a common barrier applying Jupyter notebooks to Scientific problems.
We provide code fragments in a data wrangling library that support the importing of several file types directly for Downstream utilization as well as for the purpose of importing the data into numpy, pandas, and several of the more recently developed programs that is support multidimensional data structures.

After the inputting of data, it needs to be displayed in a tabular format for inspection that to check that it was properly imported and also to carry out basic summarization statistics by column and row.

After the data have been verified as being properly imported, there has often the need to explore that data by plotting it to detect relationships between the parameters of a model and the output.
We strove to focus on the matplotlib library for the purpose of generating a wide diversity of plots.
The code fragments that we develop cover the most commonly used plots such as Scatter Plots, bar graphs (including horizontal bar graphs), kenisty density fitted distributions, heat Maps, pie charts, contour plots, and so on.
We also include examples of 3D plots.
We include a variety of variance in terms of the for batting of the tick marks and access labels as well as the keys and the form of the wines so bad users can use this information as templates to generate plots for their own purpose.
The generation of plots with lines of different shape in terms of whether they are solid or have dashes are dotted or have combinations thereof is essential because plots generated with just color blinds are vulnerable to having that their information compromised when printed in black and white.
The latter is often done at institutions that are trying to cut costs.
Although we provide some examples from some of the higher-order plotting programs like Seaborn, we focused on matplotlib because most of the other plotting programs are built on top of it, with the exception of the interactive plotting programs.

We also support the import of external images.
This is often overlooked, but these externally derived images are often important parts of the story that is being told by the Jupyter notebook.

### Juptyer specific library

We provide cell and line magics libraries that enhance the Jupyter notebook's interaction with the rest of the Computing system.


### Interactive quizzes

We developed quizzes to improve recall of the voice commands.
These quizzes are interactive and can be run in the terminal or in Jupyter notebooks {ref}`fig:quiz`.
The latter can be saved to keep a record of one's performance on a quiz.


:::{figure} ./images/quiz.png
:label: fig:quiz
An example of an interactive session with a quiz in a Jupyter notebook.
:::


To build long-term recall of the commands, one must take the quiz five or more times on alternate days according to the principles of spaced repetition learning.
These principles were developed by the German psychologist Hermann Ebbinghaus in the last part of the 19th Century [@Ebbinghaus1885MemoryAContributionToExperimentalPsychology].
They have been validated several times by other researchers.
Space repetition learning is one of the most firmly established results of research into human psychology.

Most people lack the discipline to carry out this kind of learning because they have to schedule the time to do the follow-up sessions.
Instead, we anticipate that most people will take these quizzes several times in a half hour before they spend many hours utilizing the commands.
For example, Talon Voice has an alphabet mapped to voice triggers.
The designers of this alphabet selected single-syllable words to ease the use of the alphabet.
Recalling this kind of information can be quite difficult, but by taking the quiz 10 times, you can build up the knowledge in a single day.
Recall of the alphabet will depend on how frequently it is used subsequently.
If that use occurs on subsequent days, then recall of the alphabet will be reinforced, and it may not be necessary to take the quiz again.


### Voice In Plus

Voice In is a plug-in for Google Chrome and Microsoft Edge web browsers that uses the language model built into these web browsers.
The plug-in operates in most text areas of web pages.
These text areas include those of web-based email packages and online sites that support distraction-free writing.
These text areas also include the markdown and code cells of Jupyter notebooks.
Voice In also works in various plain text documents opened in Jupyter Lab for online writing.
Obviously, Voice In will not work in standalone applications that support the editing of Jupyter notebooks, such as the Jupyter Lab app, the nteract app, and external text editors, such as VS Code, that support the editing of Jupyter notebooks.

After Voice-In Plus is activated, it will listen for words for 3 minutes before automatically shutting down.
It is very accurate with a word error rate that is well below 2\%.
It can pick out words in spite of background ambient noise, such as traffic or bird songs.
The language model is quite robust in that dictation can be performed out without the use of an external microphone.
For example, the built-in microphone available in the MacBook Pro laptop computer is sufficient.
In contrast, other VSR software requires high-quality external microphones.
The need to use an external microphone imposes a motivational barrier.

Because of the way the system is set up to utilize the browser's language model, there is not much of a latency issue.
The spoken words' transcriptions occur nearly in real-time; there is only a minor lag.
The program can generally keep up with dictation at moderate speed for at least several paragraphs, whereas competing systems tend to quit after one paragraph.
The program tends to hallucinate only when the dictation has occurred at high speed because then it starts to fall behind.
Great care has to be taken to pronounce the first word of the sentence loudly such that it will be recorded; otherwise, this first word will likely not be recorded.
This problem is most acute when there has been a pause in the dictation.

The software does not automatically insert punctuation.
The user has to vocalize the name of the punctuation that they desire.
These also have to utilize a built-in new-line command to start new lines.
The user has to develop the habit of using this command if they write one sentence per line.
This ladder form of writing is very useful for first drafts because it greatly eases the shuffling of the order of sentences during rewriting.
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
The Voice trigger does not need a comma after it, and the text replacement can span multiple lines without adding any markup, except that internal double quotes must be replaced with single quotes.
Any capitalization in the voice trigger will be ignored and written in lowercase.
The second option involves pasting in one or more lines of pairs of voice triggers and text replacements separated by commas, as you would in a CSV file.
In this option, text replacements that span more than one line must be enclosed with double quotes.
Obviously, the internal double quotes have to be replaced with single groups; otherwise the tex replacement will be truncated at the position of the first internal quote.

:::{figure} ./images/VoiceInNewSentence.png
:label: fig:newSentence
Entering a single voice trigger and the corresponding command in Voice In Plus.
:::


The carrying capacity for the storage of voice-triggered commands is unclear.
At one point, I had over 19,000 pairs of voice triggers and the corresponding text replacements.
Scrolling through the list of these voice commands was painful because it was too long.

This problem was remedied by exporting the commands to a CSV file.
Then the stored commands on the website were cleared.
The exp[ortyed CSV file was opened in a text editor and the unneeded commands were selected and deleted.
In this case, that left about 7,000 commands.
The web page with the command library displayed could then be easily scrolled.
The practical size limit of the library is between 7,000 and 19,000 commands.

Amongst the first customizations to be made are those supporting the generation of nonfiction writing.
Users may want to install our library of English expansions to avoid the tedium of converting English contractions to their expansions.
To avoid having to say `period` and ``new line'' at the end of each sentence when writing one sentence per line, the user can develop a custom command called ``new sentence'',  which is a combination of these two built-in commands.
Likewise, the custom command ``new paragraph'' can include a ``period'' followed by two ``new line'' commands, which works well when writing with blank lines between paragraphs and without indentation at the beginning of each paragraph.
Of course, these phrases will no longer be available for use in dictation because they will be used to trigger text replacements.

The VIP documentation is integrated into the GUI that the user uses to configure VIP and carry out various tasks.

A dictation session with VIP is initiated by activating the plugin by clicking on its icon.

There is a configuration page associated with the icon through which one can select the language are and even dialect of a language.

VIP has several dozen built-in commands, some of which can be used to navigate the web page.
It operates inside of text areas in web pages/
It is possible to add custom commands in the form of text replacements/
This feature requires a paid subscription whereas the remaining features are pretty available.

The Voice triggered Snippets are stored in a exported that is compatible with export to a CSV file.
The file has two columns separated by a comma.
This file locks a line of headers to be able to columns.
The Voice trigger is located in the left column and the corresponding text replacement is listed in the right column.

Frequently, it is necessary to insert a code fragment that spans multiple lines.
This code fragment needs to be enclosed with double quotation marks if the contents of a CSV file RBD copied and pasted into the bulk ad window of the GUI.
It is not possible to use a backspace to escape internal pre-existing double quotation marks.
These pre-existing double quotes have to be replaced with single quotes.

## Discussion

The following discussion points arise during our implementation of the ASR libraries described above.
Some discussion points are rows that are in question-and-answer sessions after presentations that we have made about adopting voice computing.
Other discussion points were inspired by discussions on bulletin boards and blog posts.
We limit the discussion to the software that we have presented above.

### Independence from breaking changes in Jupyter
The Jupyter project lacks built-in support for code snippet libraries.
Instead, third parties have developed several extensions for Jupyter to support code snippets in Jupyter.
Unfortunately, changes is that occur in the core of Jupyter often break these extensions.
They have to go through the trouble of setting up specific Python environments for older versions of Jupyter that still work with their favorite extension.
This can lead to difficulties when one wants to install a more recent version of a module to be run inside Jupyter because there can be dependency conflicts.
An obvious solution to this problem would be for the developers of Jupyter to incorporate one of the snippet extensions into the base distribution of Jupyter to ensure that at least one form of support for Snippets is always available.
The use of voice-triggered Snippets overcomes these difficulties with broken extensions because these Snippets are independent of Jupyter.

### Filling gap in tab-triggered snippets with voice-triggered snippets
Voice-triggered snippets also provided an opportunity to overcome the absence of extensions for Jupyter that support tab-triggered snippets.
Tab-triggered code snippets are standard in other text editors, whereas voice-triggered snippets have yet to become widespread in standard text editors.
One advantage of Jupyter Notebooks is that they run in the browser, where several automated Speech Recognition software packages operate.
We developed our libraries for two of these packages, which differ in the steepness of their learning curves and the extent of their customizability.
We did this to meet the needs of users operating at different levels of coding skill.

### The role of AI-assisted voice computing
The dream of AI-assisted voice computing is to have one's intentions rather than one's words inserted into the document you are working on.
Our exposure to what is available through ChatGPT left us with an unfavorable impression due to the high error rate.
GitHub's copilot can be used in LaTeX to autocomplete sentences.
Here again, many of the suggested completions are accurate and require editing.
These autocompleted sentences tend to slow one down, so I think there is a zero net gain.
However, the utilization of AI assistance in writing has to be disclosed upon manuscript submission.
Some publishers will not accept articles written with the help of AI-writing assistants.
This could limit the options available for manuscript submission should one use such an assistant and have the manuscripts rejected by a publisher that accepts such assistants.

### Prior art of ASR for Jupyter lab

Several extensions have been developed for Jupyter Lab enable the use of speech recognition in Jupyter notebooks.
These packages were generally developed by individuals who lack the time to maintain the software, so these extensions are quite vulnerable to growing outdated as the Jupyter platform continues to evolve rapidly.
For example, \emph{jupyterlab-voice-control}, supports the use of custom commands and is most similar to our application of voice-triggered snippets presented here.
Unfortunately, this package is not actively maintained and does not work with Jupyter 4.1.
The robustness of our approach is that it applies one of three ASR software packages that are supported by active teams of developers and that operate within Jupyter Lab and beyond Jupyter Lab, so the effort invested in mastering the commands for any of these three ASR software will have general utility beyond Jupyter Lab,

The other packages use large language models to return the transcribed text and requested code.
The first package,  \emph{jupyter-voice-comments} \footnote{\url{https://github.com/Banpan-Jupyter-Extensions/jupyter-voice-comments}},  relies on the DaVinci large language model to make comments in Markdown files and request code fragments.
The latter is fraught with code that does not work, so it is not worth the trouble for those users who are already in the trough of disappointment in the Gartner hype cycle of the current AI craze.

This program is also fairly primitive in that you need to click on a microphone icon repeatedly, which makes the user vulnerable to repetitive stress injuries.
The second package is \emph{jupyter-voicepilot} \footnote{\url{https://github.com/JovanVeljanoski/jupyter-voicepilot }}.
Although the name of the extension suggests that it might use GitHub's Copilot, it actually uses whisper-1 and ChatGPT3.
This extension requires an API key for ChatGP3.

### Fine points about voice computing
There are several caveats to keep in mind while doing voice computing.
The first point has to do with the rate of speaking, the second point has to do with coping with incorrectly returned phrase, and the third point has to do with the requirements for a microphone.
All of these points reflect the imperfect state of the current set of available language models.
Our tips suggest how one can cope with these limitations and still gain improved productivity.

The first point to keep in mind is that the rate at which you speak is an important variable.
If you speak too slowly, your words may not be interpreted as the compound name invokes a voice trigger.
Instead, the individual words printed to the screen.
On the other hand, if you talk too quickly, you may get ahead of the language model and it may stall.
If it does not seem to be responding, it is best to restart your connection with the language model.
Due to latency issues,  which vary with the software,  you will need to vary your expectations in terms of the length of verbal discourse that you can record before the system halts.

The second point is that some language models will have a difficult time with a specific word or phrase.
This is a common experience, which is rectified by using text replacements.
A difficult-to-interpret word or phrase may cause the language model to return a series of alternate words or phrases that were not intended.
The solution to this problem is to map these alternate phrases to the desired phrase to ensure that it is returned correctly.

Invariably, some of your mappings may get invoked when not intended.
This event is rare enough of an event to be tolerated.
The large language models are not perfect, and these sorts of difficulties are still widespread.
It is expected that over the next several years the language models will improve further and that these difficulties will become less common. 
Nonetheless, the ability to make the mappings between these alternate phrases and the desired phrase demonstrates the great value of being able to use text replacements to get the desired outcome.
We showed with our transcripts generated with whisper that it is also possible to apply the text replacements to the transcript after a dictation session rather than live during a dictation session, as is the case with Voice In Plus.

The third point is that the language models vary quite a bit in terms of their requirements for an excellent microphone.
It is important to realize that the more modern language models often are able to accurately transcribe your words using just the internal microphones that come with your laptop or desktop computer.
We emphasize this point because the quality of your microphone often suggests that you need to use an external microphone or a certain kind of headphone. 
This is not a general requirement.

This expectation may vary from ASR software to software.
My personal preference is not to need to wear a headset to do voice computing.
Having access to the headset all the time and needing to wear it while dictating seem to be two barriers to carrying out dictation; such barriers can turn into excuses for task avoidance.

You can inadvertently change the case of words while dictating in Voice In Plus.
To switch back to the default case, you need to navigate to the options page and select the text transform button to open a GUI that lets you set the case.

A related problem that occasionally occurs is the inadvertent activation of other voice computing software on your computer.
For example, from time-to-time, I will say a word that resembles Siri, and the Siri app will open and ask what I want.


## Other common hazards when using voice computing

In my year of using voice control every day, I have had the following mishaps.
I accidentally recorded a conversation when someone walked into my office while I had one of the ASR software packages activated.
This led to bits of our conversation being recorded in whatever text area or document happened to be active at the time.
This means that unwanted text has to be deleted later.
This is a bigger problem when a code file or code cell is being edited.
This can lead to the inadvertent introduction of bugs that can take time to remove later.

Some ASR software may be activated upon restarting the computer.
If their state is overlooked, our speech forms the sounds from our mouth or from a YouTube video or Zoom meeting could be converted into computer commands that get executed in unintended ways.
I found that my external microphone will block my external speaker and interfere with the audio in Zoom sessions.
I also had two voice-control software on at the same time and had clashes occur between them.


## Future directions

One future direction is to build the libraries that have been developed to date.
This includes the development of a method of facilitating voice stops, in analogy to tab stops in advanced text editors, for the purpose of editing sites that have parameters or parameter values that need to be changed to customize the code snippet.
The other related advance would be the mirroring of identical voice stops.
