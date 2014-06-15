:author: Dominik Klaes
:email: dklaes@astro.uni-bonn.de
:institution: Argelander-Institute for Astronomy, Bonn, Germany

------------------------------------------------
Teaching Python to undergraduate students
------------------------------------------------

.. class:: abstract

Teaching undergraduate students in programming is interesting and challenging at 
the same time, because one has to deal mostly with two types of students: Those who have 
already experience with programming and those who have not. I will present two models from Bonn 
University for Physics students with now much more responsibility for the tutors 
and would like to initiate discussions about different systems all over the 
world.

.. class:: keywords

   Python, teaching, undergraduate, students, physics, astrophysics


Introduction
------------

In many studies one needs sooner or later a computer which can help with or solve a certain problem. Students starting their studies at a University do not have necessarily experience with a computer beyond mailing and chatting. To ensure that especially those students can catch up and also those, who have already more experience in e.g. programming, can still learn something, there exist a course called \``Introduction to computer science and programming'' for Physics students at the University of Bonn. Later in their studies, students need several aspects of this course again, especially:


1. How to find help / information
2. Basic Unix knowledge
3. LaTeX
4. Plotting
5. Data fitting
6. Programming and debugging


In the past, this course, which takes place every two semesters, consisted of two part: A weekly 1-hour-lecture and a weekly 2-hours-exercise class. Due to the experience of 6 years, so also including two \``sets'' of Bachelor in Physics (3 years) and Master in Physics or Astrophysics (2 years) students, and changing small aspects already in between, it became clear that this course has to be changed completely. In the following, I will explain the basics of our program, so that the changes we made will become more reasonable. 


Basics
------

The Bachelor in Physics (duration: 6 terms, 3 years) and the Master in (Astro)Physics (duration: 4 terms, 2 years) both require a lot of computer knowledge for different aspects. Both programs consist of so called \``modules'' which normally cover one topic of the education. These modules were introduced in Europe through the \``Bologna Process'' (also known as \``Bologna Reform'') in 1999 which should in principle allow the students to change the university during their studies very easily. This required some changes in the courses themselves and also some \``generalized'' calculations of the workload of individual courses so that they can compared to each other, especially between different countries, where the university systems differ significantly. Besides this module aspect, there are two more important properties I will use later: the \``credit points'' and the \``SWS''.

The first one reflects the average total amount of time a student has to raise for a module including lecture time, exercise class time, time to solve exercise sheets and also preparing for the final exam(s), which is normally a written or oral exam, a presentation, a report or something similar. It was defined that 1 credit point (CP) is equivalent to 30 hours work. This also allows to weight different modules so that a module with more credit points (so more work) can be weighted more in the final, averaged grade for the Bachelor grade. Every semester is such designed that about 30 CPs per semester are obtained.

The last one gives the number of hours per week (German: **S**\ emester\ **w**\ ochen\ **s**\ tunden or short SWS) the student is at the university, so normally lecture and exercise class time. This allows the student to judge how much time he or she has to spend at the university and how much time has to be used to prepare the lectures / exercises and rework the materials.

More information can be found on the official website (http://www.ehea.info/) or, for a short overview, also on Wikipedia (http://en.wikipedia.org/wiki/Bologna_Process).


Figure :ref:`fig:overview` shows the overview of the Bachelor in Physics program at Bonn University (taken from [Bachelor]_ with permission of the examination board and translated). It is split up into several categories and shows when which module is normally taken and assumes that the students passes all exams. If not, the course can normally repeated one year later. The module of interest is located in the first semester in the category \``Extra courses''.

.. figure:: bsc2_grafik_englisch-rotated90.pdf
   :align: center
   :figclass: w
   :scale: 80%

   Overview of the Bachelor of Science in Physics program at the University of Bonn. :label:`fig:overview`


Usage of skills
---------------

This part will explain which skills are needed during the studies and are obtained during the EDP module. For this purpose, the numeration used in the introduction is used.


1. How to find help / information: One might assume that high school students know how to get additional information for a certain problem using e.g. Google which they used for homeworks or presentations. While some students do not have any problems with using such a search engine, the others do not know how to use it efficiently, e.g. showing only the newest search hits or from a certain website or excluding specific search words. This then increases the time students need for their self-studying at all stages of their studies: looking up formulas, checking the syntax of a command or looking for the solution of a (programming) problem. One can say that for undergraduate students and their problems there is at least another person how already had the same or similar problem and provides a solution. On the other side this reduces the ability and/or motivation to think first about a problem so a compromise has to be found.

2. Basic Unix knowledge: Most of the first year students have only basic knowledge of an operating system, normally Windows. In (Astro)Physics Unix operating systems, normally Linux, are mostly used. In the last decade, Linux and its usage has improved a lot and for GUI users there are only a few small differences for basic Windows users which simplifies the usage and the first contact for those students. Also here the first programming skills can be obtained even without being noticed by the students by writing small shell scripts.

3. LaTeX: LaTeX is widely used in the (astro)physics community to write scientific papers, thesis or presentations. In this Bachelor program the students can use their learned knowledge directly at the end of this course by writing the report (including several tasks from the different topics). It is compulsory the hand in the report using LaTeX. During their further studies they will use it during the laboratory courses (writing the reports), the presentation seminar where they learn how to create a good presentation (here LaTeX is optional, other programs are allowed to use) and finally their Bachelor thesis (LaTeX is not mandatory but especially when using formulas and references, LaTeX shows its enormous power). Especially in the beginning the students do not understand why one should use LaTeX because programs like Microsoft Word and PowerPoint or LibreOffice Writer and Impress do a sufficient job for the problems they had up to now. Especially the error messages and/or warnings LaTeX produces are not always easy to understand and even if, not directly solvable for someone without experience. Since LaTeX can be interpreted somehow as programming, and so code debugging is important, also the last point (\``Programming and debugging'') will become relevant again.

4. Plotting: *A picture is worth a thousand words.* This saying is also valid for a lot of numbers one has to deal with, e.g. from a measurement series. While this is well known, how to get from numbers to a \``good'' plot, so e.g. readable and understandable, is not easy at all. This starts with choosing the \``right'' program: A simple one like e.g. gnuplot, a tool like Mathematica or Maple, or directly a programming language like C + ROOT or Python + matplotlib? Also questions like style (color or b/w plot, line style, labels etc.) and formats (JPEG, (E)PS, PNG etc.) including differences have to be made clear. Only being aware of this guarantees up to a certain point \``good'' plots that everyone can read and understand. A quite recent example for a \``bad'' plot and font style is the Comic Sans font style from the CERN higgs boson discovery slides in 2012 [ComicSans]_.

5. Data fitting: Having only data does not help anyone, the data has to be interpreted. Part of this is fitting a law or a formula to the data. For first year Bachelor students it is difficult to understand what \``data'', including e.g. errors, is and how to work with it because this is normally not taught in high school or if it is taught, only basics are available. This knowledge is espicially used during the laboratory courses and if applicable during the Bachelor thesis. Handling, fitting and interpreting data is one of the basic tool that a scientist should be able to use. Furthermore knowledge about the different ways to fit data (unweighted, weighted) or different methods (e.g. :math:`\chi ^{2}`) is essential, not only for laboratory courses.

6. Programming and debugging: Most of our first year Bachelor students do not have any experience with programming. As already explained before, programming has become very important over the last decade and is required nowadays in (Astro)Physics. Unfortuanely there is not **the** programming language that contains all needed tools, is easy to learn and use, compatible with many different computer architectures and so on. Also looking into the different working groups does not solve the problem: Currently languages like C/C++ and Python are very common but also Fortan and Perl are still used, especially because older progams are written in these languages and it would take a lot of time and man power to translate the code into a \``newer'' language and to test it. Compared to the other already mentioned topics, this part could only be broached, which means in terms of time only 1 to 1.5 out of 12 lectures and exercise classes at the end of the lecture time when also almost all students concentrate more on the exam preparation.


Past vs. present
----------------

Starting in 2006, the EDP module has evolved and improved over the last years. In the beginning this course consisted of a weekly one-hour-lecture and a weekly two-hours-exercise class. The lecture was used to present the theoretical background, followed by a live demonstration. Since this lecture took place in a lecture hall and not in the CIP pool, only a few students had a computer (their laptop) which they can use to directly repeat the shown examples. Furthermore questions that can occur while trying out the examples could not be raised directly. This also means that the lecturer gets less feedback to improve the lecture and its style. One consequence was also that less and less students attended the lecture since there is no compulsory attendance in our lectures. All this made the lecture more or less obsolete but cannot be dropped due to regulations so that all problems and questions were shifted into the exercise class and the tutor had to deal with them. For this, two hours per week is not enough, especially because the students prioritize their work and besides a mathematics and an experimental physics lecture, this lecture and exercise class appears not very important for them. Influences on the time spent on preparation are obvious. To do the splits between the regulations for the Bachelor, the workload for the students and tutors and the efficient time usage, it was tested to switch to one \``main'' lecture every few weeks for each \``main'' topic such as Linux, LaTeX and programming, and a weekly three-hours-exercise class. This enhances the possibility for the students to see how to solve a certain problem, test this solution on their own and, if there are questions left, directly ask the tutor who can directly help and give individual advices. In this solution, the tutors have much more responsibility for the education of the students and in times when it is sometimes complicated to get tutors at all, a complicated issue.

test


Why Python and not C?
---------------------



References
----------
.. [Bachelor] http://tiny.iap.uni-bonn.de/mhb/bsc_grafik.pdf
.. [ComicSans] http://www.buzzfeed.com/babymantis/cern-uses-comics-sans-to-explain-higgs-boson-1opu