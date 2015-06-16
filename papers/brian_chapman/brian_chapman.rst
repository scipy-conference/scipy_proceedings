:author: Brian E. Chapman, Ph.D.
:email: brian.chapman@utah.edu
:institution: Department of Radiology, University of Utah

:author: Jeannie Irwin, Ph.D.
:email: jeannieirwin@gmail.com
:institution: Unaffiliated


---------------------------------------------------------------
Python as a First Programming Language for Biomedical Scientist
---------------------------------------------------------------

.. class:: abstract

We have been involved with teaching Python to biomedical scientists since 2005, In all, seven courses have been taught: 5 at the University of Pittsburgh, as a required course for . Student have primarily been biomedical informatics graduate students with other students coming from human genetics, molecular biology, statistics, and similar fields. The range of prior computing experience has been wide: the majority of students had little or no prior programming experiences while a few students were experienced in other languages such as C/C++ and wanted to learn a scripting language for increased productivity. The semester long courses have followed a procedural first approach followed by an introduction to object oriented programming. By the end of the course students produce an independent programming project on a topic of their own choosing. 

The course has evolved as biomedical questions have evolved, the Python language has evolved, and online resources have evolved. Topics of primary interest now focus on biomedical data science with analysis and visualization using tools such as Pandas, scikit-learn, and Bokeh. Class format has evolved from traditional slide-based lectures supplemented with IDLE programming demonstrations to flipped-classrooms with IPython notebooks with an active learning emphasis. Student evaluations indicate that students tend to find the class challenging, but also empowering. The most difficult challenge for most students has been working with their computers (installing software, setting environment variables, etc.) Tools such as Canopy, Anaconda, and the IPython notebook have significantly reduced the extraneous cognitive burden on the students as they learn programming.

In addition to reviewing the nature of the course, we will review the long-term impact the course has had on the students, in terms of their retrospective evaluation of the course and the current nature of their computational toolbox. We will also discuss how our experience with these courses has been leveraged in designing a Python-centric summer-school for biomedical data science.

.. class:: keywords

   education, biomedical informatics, biomedical sciences

Introduction
-------------------------

Python has become the most popular language for majors at the top computer science departments (`Philip Guo, "Python is Now the Most Popular Introductory Teaching Language at Top U.S. Universities" <http://cacm.acm.org/blogs/blog-cacm/176450-python-is-now-the-most-popular-introductory-teaching-language-at-top-us-universities/fulltext>`_). Motivations for Python as the first language are its simple semantics and syntax [Stefik2013], leading to students making fewer mistakes, feeling more confidence, and having a better grasp of programming concepts relative to peers taught with more traditional, lower-level languages such as C or Java [Koulouri2014]. Since Python is a multi-paradigm programming language, it offer great pedagogical flexibility. Python is also an active language with many open source projects and employers looking for Python programmers (`"Is Python Becoming the King of the Data Science Forest?" <(http://www.experfy.com/blog/python-data-science/>`_, `"The RedMonk Programming Language Rankings: January 2015" <(http://redmonk.com/sogrady/2015/01/14/language-rankings-1-15/>`_).

These same characteristics make Python well-suited for teaching programming to students without a computational background. The biomedical sciences are increasingly becoming computationally oriented. The installation of electronic medical records, digital public health registries, and the rise of -omics (e.g. genomics, proteomics, biomics) means biological discovery and healthcare delivery increasingly requires the storage and analysis of digital data. That is, these disciplines increasingly require the application of computation. However, students in biomedical sciences largely arrive in graduate school without computational skills. For example, biomedical informatics graduate students have diverse backgrounds including medicine, nursing, library science, psychology, and linguistics. In order to be successful in their graduate studies, the students must quickly pick up programming skills relevant to the biomedical problems they are addressing in their graduate studies.

Rather than asking students to take multiple undergraduate computer science courses, we have designed a one semester Python programming course that allows the students to develop programming skills for their graduate studies. In this paper we first provide a brief summary of the course. Using both our personal observations and surveys of past teaching assistants and students, we then summarize our experiences with this course over the past ten years. Finally, we provide suggestions for future teaching of biomedical graduate students.

Course Objectives
-------------------------

The course we describe here was originally created as a required programming course for biomedical informatics students at the University of Pittsburgh. Most recently it has been offered at the University of Utah as a required course for an applied genomics certificate and as an elective for a variety of biomedical science graduate programs including biomedical informatics, biology, human genetics, and oncological science. One of us (BEC) has seven years experience as the course designer and instructor, the other (JI) has one year experience as a student (with a prior instructor) and four years experience as a TA with BEC at the University of Pittsburgh. 

As we conceive the course, it has two intended purposes. First, the course is intended to provide the students sufficient programming experience that they can use programming in their graduate studies, meaning they should be able to 

a. continue to learn and improve their Python programming skills on their own, 
b. successfully use Python in other programming-oriented courses during their graduate studies, 
c. use Python in their thesis or dissertation work, 
d. use their experience with the Python course to teach themselves another programming languages as needed. 

Second, the course was intended to introduce students to the nature of biomedical data: what it looks like, what some of the standards associated with it are, and how to represent and model the data. For example, with clinical lab values, students would be asked to address whether integers, floating point numbers, or strings would be most appropriate for representing the depicted values, what type of meta-data  should be associated with the value (e.g. units, method of measurements), and what sort of data structure would be most appropriate to store the data and meta-data (e.g. list, tuple, or dictionary).

Simultaneously, the course tried to illustrate biomedical problems that researchers were currently addressing. Through this it was hoped that students were not learning programming in a vacuum or purely abstractly but in the context of problems in their fields.

The course was described to the students as a “boot camp” to get students with little or no programming experience up to speed for starting their graduate work. Consequently, as a "boot camp" the students should expect to spend more time than in an average three-credit course. Because this course was viewed as the foundation for subsequent graduate studies that the student had voluntarily enrolled in, it was assumed that the students wanted to learn and consequently were more interested in learning than in the grade received in the course. 
>

The course was taught with a more empirical than theoretical approach, using the Python (and IPython) shell to try out code snippets to see what happens. We occasionally quoted Ms. Frizzle from The Magic School Bus: "Make mistakes, get messy" (CHECK ACTUAL PHRASE).

First taught in 2005, the nature of the course has transformed as the available computational and pedagogical tools have expanded. For example, learning how to read files with Pandas has replaced exercises in reading and parsing comma separated files using low-level Python functionality. Similarly, static slides have been replaced by interactive IPython/Jupyter notebooks and YouTube videos. 

Course Structure
-------------------------

The course is structured around weekly homework assignments and a course project. Additional features have included quizzes (scheduled and pop), in-class and take-home exams, peer code review, in-class individual, pair, and group and class-wide programming assignments. Homeworks were designed to both reinforce topics that were covered in class and to require students to learn additional material on their own, primarily in the form of finding (with hints) and using modules within the Python standard library. Course projects were intended to have the student focus on an area of interest to themselves, require them to learn additional tools, and to require them to integrate various topics covered in class. For example, they must define a base class and inherited class, interface with a database (e.g. SQLite), and have some sort of graphical user interface (e.g. IPython notebook, TKinter, Flask, Django).

The semesters were roughly split in half. In the first half-semester, the course covered the fundamentals of imperative programming including numeric and string manipulation, if/else, while/for, functions, and classes. Homework assignments became progressively more demanding. In the second half-semester, topics learned in the first-half semester were reinforced through exploration of various Python packages. Homeworks were decreased to allow the students more time to focus on their term projects. Because these illustrative applications are somewhat arbitrary, the students could request/select which topics are covered. 

In-class lectures were minimized in favor of interactive programming assignments, either in the form of class-wide, small group, or individual programming projects, code reviews, or discussions about sticking points encountered during the homework. To ensure that students were motivated to be prepared for class, a "random student selector" was used to determine who would be at the podium for the next explanation or problem.

Students were encouraged to work together on homeworks and optionally could work together on term projects.


Evaluations and modifications
-----------------------------

We reviewed previous course materials and end-of-course student evaluations. In addition we solicited input from past teaching assistants and sent a questionnaire to previous students to try to better assess the long-term usefulness of the course. The questionnaire was generated using SurveyMonkey and consisted of a combination of multiple choice, Leikart scale, and free response questions.  Past course lists were obtained from the University of Pittsburgh and the University of Utah. Where current e-mails were not known from the University, connections were sought through LinkedIn and other social media connections. Previous teaching assistants for the courses were e-mailed directly. Course materials were reviewed to observe changes in content over the years.red around. Previous teaching assistants for the course were solicited for their analysis of the course. Twenty-four previous students responded to the survey. However, one of the responses was blank on all questions, and so we will report results based on 23 responses. 


Course Experience
-------------------------

One of the greatest challenges we have observed in teaching is the lack of basic computer skills among students. The initial challenge of installing Python, an appropriate code editor/and or an integrated development environment, getting environment variables set, etc. has been significantly diminished by the use of third-party, complete Python installations, such as Anaconda or Canopy. In the retrospective student survey 10 respondents said they would like to have been taught how to work in computer shells prior to beginning instruction in programming. (In a related response, 6 would have liked to have been taught UNIX/Linux skills prior to beginning instruction in Python.)

One of the problems that we have observed repeatedly is a lack of general problem solving skills among students. This is immediately manifested in the difficulty of teaching debugging skills but more generally in how to model a problem. Five of the respondents listed being taught general problem solving skills prior to learning programming would likely have been beneficial.

Another challenge with teaching the class has been access to biomedical data with which to apply programming. There are now several publicly available medical data sets available. NCI biomedical imaging arhciveI, MT Samples, MMIC. A variety of -omic datasets are now publicly available, largely due to NIH data sharing requirements connected to funding. Nonetheless, availability of large, rich data sets remains a limitation for the dual purpose of the class.



Prior Student Experience
---------------------------------------------------------------

Roughly one-third of respondents described their application area at the time of the course as data analysis, another third described themselves as working in bioinformatics, and the rest were split among areas such as task automation, natural language processing, and scientific programming.

Roughly equal numbers of students reported themselves as having no prior programming experience (39%) or some prior programming experience (e.g. one prior programming course, self-taught exploration) (35%). Twenty-six percent reported moderate programming experience. No students reported extensive programming experience. Biomedical informatics  generally required a prior programming course as a pre-requisite for admission. Roughly equal number of students reported prior experience with C/C++ (9) and Java (8).

Although these responses are anonymous, and we do not know which responses correspond to which students, as an instructor BEC did not see a noticeable difference in class performance between students with no and some prior experience. However, at least one TA felt strongly that prior experience was necessary for success in the course. Acknowledging that the course is certainly easier for someone with prior programming experience, it is not uncommon for a student with no prior programming experience to be the top performing student in the course.


Suggested Pre-programming Instruction
---------------------------------------------------------------

In the survey we asked the students "If you were redesigning the Python course, what skills/knowledge sets would you like to have been taught prior to starting programming instruction?" Seventeen our of 22 respondents selected familiarity with shells/command line, nine selected familiarity with Unix/Linux, and seven selected general problem solving skills. 

These responses affirm our own experience that the greatest barrier to the students' success is lack of basic computer skills. It should also be noted that the survey was only sent to students  who had completed the course. Anecdotally a large number of students dropped the class before programming really began simply out of frustration with trying to install Python and text editors, set environment variables, etc. This was especially true of Windows users. In the most recent class, we used git for homework, and Windows users almost all adopted the Git shell as their default shell rather than the Windows terminal.

Lack of general problem solving skills was a persistent problem that manifest itself with difficulty in debugging, but also in tackling open-ended problems. Students struggled with how to break a problem into small parts, or how to start with a partial solution, test it, and then move on to a more complete solution.


Students Retrospective Review of Course Value
---------------------------------------------------------------

Sixteen of the 23 respondents agreed or strongly agreed that "programming is an integral part of my professional work." One respondent strongly disagreed and two disagreed.

Thirteen out of 23 respondents agreed or strongly agreed that "Python is my primary programming tool." 19 out of 23 respondents agreed or strongly agreed that "Learning Python was valuable for helping me subsequently learn additional programming language(s)." Only one respondent disagreed. 

As an overall assessment, 22 out of 23 students agreed or strongly agreed that "learning Python was valuable for my career development." It is important to note that this includes people that listed themselves as having extensive prior programming experience and also respondent who do not view programming as part of their professional activity. 

Reasons people listed for not using Python after the class included limitations of the language (memory management, speed), not considering it a statistical language (as compared to R), and collaborators using other languages (Java, Perl).

Python for a Data Science Boot Camp
---------------------------------------------------------------


The Python course needs to be part of a larger series of courses. First, the students need to be introduced to working with the shell. One of the stated advantages of Python is that it can be run on multiple platforms. However, it can be argued that the majority of scientific computing occurs on a Linux platform. Having learned to work in a Linux shell, it would be relatively straight forward to learn to work in the Windows terminal. However, to learn how to work in Linux would require students to learn how to use virtualization tools, adding more and more complexity to the students start. To address this we are building an on-line, computational learning environment based on GitLab, Docker, and the Jupyter notebook. The Terminado emulator in the notebook will be used to help students learn Linux shells. This environment will be used for a summer biomedical data science boot camp for clinicians and others without a computational background. Python will be used as the programming language. As discussed here, the Python program course, similar to what is described here, will be preceded by mini courses on working with Linux shells and problem solving. Following the programming course, there will be short courses on visualization, statistics, and machine learning, also using Python. The plan is for the boot camp to feed into various computationally oriented biomedical graduate programs.


Summary and Conclusion
---------------------------------------------------------------


While including a range of responses---some students have hated the course---end-of-course student evaluations primarily reflect the fact that the course is challenging but useful. We suspect that the responses to the survey are someone biased in that we were more likely to have current e-mail addresses for students who had gone on to academic or high-profile industry positions. Also the retrospective survey responses are more positive on average than end-of-course evaluations, although this could also an improved assessment over time by the students. In personal communications, two students referred to the course as  "life changing" and "liberating." The student who called the course life changing left medical practice and received a graduate degree in biomedical informatics from Stanford University. The student who called the course "liberating" had taken the programming course towards the end of her graduate studies. Once she learned how to program she left liberated from dependency on her advisor and his staff for conducting her graduate work.

In open responses in our survey, former students expressed a variety of ways Python has helped them. In addition to expected comments about increased personal productivity and confidence, one former student who does not program as part of his professional responsibilities noted how valuable the class was him for future work with programmers. As prevalent is programming now is in biomedical sciences, we believe this illustrates how broadly valuable the course can be.

In summary, we believe that Python is an excellent choice for teaching programming to graduate students in biomedical sciences, even when they have no prior programming experience. In the course of a semester, students are able to progress from absolute beginners to students tackling fairly complex and often useful term projects.

References
----------
.. [Koulouri2015] T. Koulouri, et al. *Teaching Introductory Programming: A Quantitative Evaluation of Different Approaches*,
           Trans. Comput. Educ., 14(4):1---26, December 2014.
.. [Stefik2013] A. Stefik and S. Siebert. *An Emperical Investigation into Programming Language Syntax,* Trans. Comput. Educ., 13(4):1---19, November 2013.


