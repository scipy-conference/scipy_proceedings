.. #############################################################################################################
.. #############################################################################################################

.. raw:: latex

   \newcommand{\PadA}       {{\color{white}{.}}\\}
   \newcommand{\Label}   [1]{{\raisebox{-1pt}{\includegraphics[width=8pt]{ref-#1}}}}
   \newcommand{\SPMtok}  [1]{{\raisebox{-2.5pt}{\includegraphics[scale=0.9]{tok-#1.pdf}}}}
   \newcommand{\SPMtokB} [1]{{\raisebox{-0.5pt}{\includegraphics[scale=0.9]{tok-#1.pdf}}}}
   \newcommand{\snippet} [1]{
                              {\color{white}{.}}\\
                              \centerline{\includegraphics[scale=1.0]{#1}}\\
                            }
   \newcommand{\snippetB}[1]{
                              {\color{white}{.}}\vspace*{8pt}\\
                              \centerline{\includegraphics[width=3.5in]{#1}}\vspace*{8pt}\\
                            }
   \newcommand{\snippetC}[1]{
                              {\color{white}{.}}\vspace*{8pt}\\
                              \centerline{\includegraphics[width=3.5in]{#1}}\vspace*{-12pt}
                            }

.. #############################################################################################################
.. #############################################################################################################

:author: Minesh B. Amin
:email: mamin@mbasciences.com
:institution: MBA Sciences, Inc

.. #############################################################################################################
.. #############################################################################################################

-------------------------------------------------------------------------
A Technical Anatomy of SPM.Python, a Scalable, Parallel Version of Python
-------------------------------------------------------------------------

.. class:: abstract

   SPM.Python is a scalable, parallel fault-tolerant version of the
   serial Python language, and can be deployed to create parallel
   capabilities to solve problems in domains spanning finance, life
   sciences, electronic design, IT, visualization, and
   research. Software developers may use SPM.Python to augment new or
   existing (Python) serial scripts for scalability across parallel
   hardware. Alternatively, SPM.Python may be used to better manage
   the execution of stand-alone (non-Python x86 and GPU) applications
   across compute resources in a fault-tolerant manner taking into
   account hard deadlines.

.. class:: keywords

   fault tolerance, parallel closures, parallel exceptions, parallel invariants,
   parallel programming, parallel sequence points, scalable vocabulary,
   parallel management patterns

Prologue
--------

   Consider the following acid test for general purpose parallel
   computing. A serial session is depicted on the left, whereas
   the session on the right describes its parallel equivalent:

.. raw:: latex
   
   \centerline{
   \begin{tabular}{c}
   \vspace*{-4pt}\\
   \includegraphics[width=3in]{snippet-prologue.pdf}\\
   \vspace*{-8pt}\\
   \end{tabular}
   }

For

.. raw:: latex

    example, the command \SPMtok{cmdA} may be a
   parallel make-like capability, while the command \SPMtok{cmdB}
   may be a map-reduce capability. At the same time, the
   command \SPMtok{cmdC} may be a fine grain parallel
   SAT solver that limits itself to resources with specific incarnations of
   those utilized by the command \SPMtok{cmdA}. Finally, \SPMtok{cmdD}
   may be a parallel graph-based analytics capability.

   Yet, notwithstanding the prosaic serial session, the equivalent
   parallel session is in fact predicated on solutions to what were
   several formally open problems, including (a) defining a scalable
   vocabulary rich enough to capture the essence of a wide range of
   parallel problems, (b) the ability to utilize a collection of
   hardware resources in completely different ways, depending on the
   nature of parallelism exploited by the respective commands within
   the same session, and (c) the ability to treat the conclusion of
   each parallel command as a sequence point, thus guaranteeing that
   there would be no pending side effects post conclusion.

Introduction
------------

.. raw:: latex

    In this paper, we shall review (patented) SPM technology, and the
    methodology behind it, both predicated on the supposition that parallelism
    entails nothing more than the {\em management} of a collection of {\em serial tasks},
    where {\em management} refers to the policies by which:
    \begin{itemize}
    \item tasks are scheduled,
    \item premature terminations are handled,
    \item preemptive support is provided,
    \item communication primitives are enabled/disabled, and
    \item the manner in which resources are obtained and released
    \end{itemize}
    and {\em serial tasks} are classified in terms of either:
    \begin{itemize}
    \item Coarse grain – where tasks may not communicate prior to conclusion, or
    \item Fine grain – where tasks may communicate prior to conclusion.
    \end{itemize}

    We
.. raw:: latex

     shall review how SPM.Python augments the serial Python language
    to include a suite of parallel primitives, henceforth referred to
    as parallel closures. These closures represent the sole means by
    which to express any parallelism when leveraging SPM.Python. Their
    APIs are designed to be as close to the developer's intent as
    possible, and therefore easy to relate to.  Furthermore, the API
    of all closures represent the boundary that delineates the serial
    component (authored and maintained by the developer) from the
    parallel component (authored and embedded within SPM.Python).

    Specifically, the context for and solutions to four formerly
    open technical problems will be reviewed:
    (a) decoupling tracking of resources from management of resources,
    (b) declaration and definition of parallel closures, the building blocks of all parallel constructs, 
    (c) design and architecture of parallel closures in a way
          so that serial components are delineated from parallel
          components, and
    (d) extensions to the general exception handling infrastructure
          to account for exceptions across many compute resources.

    We will illustrate key concepts by reviewing a simple, scalable,
    fault-tolerant, self-cleaning 60-line Python script that can be
    used to launch any stand-alone (x86 or GPU) applications in
    parallel.  Appendix A will provide another self-contained Python
    script that calculates the total number of prime numbers within a
    given range; thus, illustrating how any Python module may be
    parallelized using one of SPM.Python's several built-in parallel
    closures.

.. raw:: latex

   \clearpage\newpage

.. #############################################################################################################
.. #############################################################################################################

.. raw:: latex

   \begin{center}
  %\vspace*{0.5cm}
   \begin{figure}[hbt]
   \noindent{\hfill\includegraphics[width=2.8in]{fig-softwareStack.pdf}\hfill}
   \caption{In order to facilitate the exploitation of multiple, potentially
   different, forms of parallelism within a single session of SPM.Python,
   tracking of resources is decoupled from the management of resources.
   Therefore, while the tracker is always online, at any moment in time, at most one task manager may be online. \DUrole{label}{fig-softwareStack}}

   \vspace*{0.4cm}
   \noindent{\hfill\includegraphics[width=2.05in]{fig-ppdefinedeclare.pdf}\hfill}
   \caption{Parallel sequence points in terms of online and offline states of
   the Hub and Spokes. On the Hub, transition to online occurs when a task manager is invoked;
   transition back to offline occurs when the said manager concludes.
   On the Spoke, transition to online occurs when a task evaluator is invoked; transition back
   to offline occurs when the said evaluator concludes. \DUrole{label}{fig-ppdefinedeclare}}
   \end{figure}
   \end{center}\vspace*{-26pt}

Related Work
------------

.. raw:: latex

   Traditionally, most parallel solutions in Python have taken the
   form of: (a) distributed task queues like Celery\cite{Celery},
   Parallel Python\cite{PPy}, (b) distributed frameworks like Disco
   (MapReduce)\cite{Disco}, PaPy (parallel pipelines)\cite{PaPy}, or
   (c) low-level wrappers around HPC libraries like MPI\cite{MPI},
   PVM\cite{PVM}. In sharp contrast, SPM.Python is a single runtime
   environment that provides access to multiple different, distinct
   forms of parallelism by way of parallel primitives called parallel
   closures.. Furthermore, these closures are architectured to be as
   close to the developer's intent as possible -- in terms of, say,
   either coarse or fine-grain DAG/templates/hybrid flows, and lists
   -- while de-emphasizing low-level error-prone concepts like locks,
   threads, pipes, mutexes and semaphores.\vspace*{-10pt}

Tracking of Resources
---------------------

.. raw:: latex

   In SPM.Python, compute resources are tracked independently of any
   task manager. In operation, any task manager may come online and
   request resources from the tracker. The task manager would then
   manage the execution of tasks using the acquired resources, and
   when done, go offline (i.e. release the resources back to the
   tracker).  Another task manager may subsequently come online,
   obtain the same or different resources used by a previous task
   manager, and utilize those resources in a completely different
   way. In other words, the task managers can be implemented more simply
   because each manager would have a more narrowly focused discrete
   policy. Furthermore, a tight coupling can be established between a task
   manager and the communication closures, thus preventing a whole
   class of deadlocks from occurring. More details can be found at
   \cite{USP01}.\vspace*{-10pt}

Declaration and Definition of Parallel Closures
-----------------------------------------------

.. raw:: latex

   In SPM.Python, parallel closures are the building blocks of all
   parallel constructs, and provide the sole means by which one may
   express how serial components interact with parallel
   components. The interactions may take place in one of two contexts
   (a) when creating, submitting, and evaluating tasks, and (b) when
   creating and processing messages. However, any usage of a parallel
   closure within any resource is predicated on a successful, safe,
   asynchronous and race-free declaration and definition across many
   compute resources. We solve this problem by augmenting the
   traditional concept of serial sequence points by introducing the
   notion of {\em offline} and {\em online} states. The declaration
   and definition of parallel closures is only permitted when the
   resource in question is in the {\em offline} state -- a state when
   SPM.Python guarantees that the serial component of the resource may
   not communicate with the outside world and vice versa. So, all
   resources start off {\em offline} (\Label{A}, \Label{C}).

   On
.. raw:: latex

    the Hub, the transition to the {\em online} state occurs when a
   parallel (task manager) closure is invoked; the transition back to
   the {\em offline} state does not occur until just before the
   closure concludes. On the Spoke, SPM.Python receives a task from
   the Hub while {\em offline} (\Label{C}), and at which point any
   preloading of Python modules is performed.  One side effect of this
   preloading may be the declaration and definition of parallel
   closures. Next, the transition to {\em online} is made before
   SPM.Python invokes the callback (\Label{D}) for the task; the
   transition back to {\em offline} does not occur until just after
   the callback concludes.

.. raw:: latex

   \clearpage\newpage

.. #############################################################################################################
.. #############################################################################################################

.. raw:: latex

   \begin{center}
   \vspace*{0.5cm}
   \end{center}

.. figure:: fig-ppcoarsegrainPy.pdf
   :width: 3.5in
   :class: align-center
   :figclass: hc

   The architectural and runtime perspectives of coarse grain task manager closures.
   Note that such closures do not permit tasks to communicate prior to conclusion.
   :label:`fig-ppcoarsegrainPy`

.. figure:: fig-ppfinegrainA.pdf
   :width: 3.5in
   :class: align-center
   :figclass: hc

   The architectural and runtime perspective of fine grain (limited) task manager closures.
   Note that such closures permit tasks to communicate only with the Hub.
   :label:`fig-ppfinegrainA`

.. figure:: fig-ppfinegrainB.pdf
   :width: 3.5in
   :class: align-center
   :figclass: hc

   The architectural and runtime perspective of fine grain (general) task manager closures.
   Note that such closures permit communication among Spokes and, if appropriate, with the Hub.
   :label:`fig-ppfinegrainB`

.. raw:: latex

   \begin{center}
   \vspace*{0.5cm}
   \end{center}

Types of Fault-Tolerant Parallel Closures
-----------------------------------------

.. raw:: latex

    A key tenet of the serial software ecosystem is the asymptotic parity
    between the serial compute resources available to the developers and
    the end-users, which makes possible the reporting, reproduction, and
    resolution of bugs.

    With parallel software, this most fundamental of tenets is violated;
    software engineers need to be able to produce high-quality parallel
    software in what is an essentially serial environment, yet be able to
    deploy the said software in a parallel environment.

    SPM.Python addresses this dichotomy by offering a suite of
    easy to relate to parallel closures. These closures enable the
    prototyping, validation, and testing of parallel solutions in an
    essentially serial-like development environment, yet are scalable when
    exercised in any parallel environment.\vspace*{-10pt}

Coarse grain
------------

.. raw:: latex

   Exploiting coarse grain parallelism is anchored around the
   asynchronous declaration and definition of a parallel (task
   manager) closure (\Label{macro}) across all resources (Hub and
   Spokes).  On the Hub, this is depicted by (\Label{A}).  On the
   Spokes, this is only possible prior to the evaluation of a task, as
   depicted by (\Label{C}), when the modules may be preloaded.

   Next,
.. raw:: latex

    existing serial functionality (\Label{serialPy}) may be parallelized
   by having it be augmented with serial code (\Label{task}) to:
   \begin{itemize}
   \item[$\bullet$\hspace*{0.1cm}] generate and submit tasks to the parallel
   task manager, and handle status reports/exceptions
   from tasks, as depicted by (\Label{B})
   \item[$\bullet$\hspace*{0.1cm}] evaluate tasks, as depicted by (\Label{D})
   \end{itemize}
   Finally, actual parallelism can commence by invoking the
   task manager on the Hub with a collection of tasks, and a handle to a
   pool of resources (\Label{B}). The backend of the task manager would
   ensure the concurrent scheduling and evaluation of tasks across all Spokes.
   Note that coarse grain task manager closures do not permit
   the usage of any form of communication closures (\Label{microDisabled}).\vspace*{-10pt}

Fine grain (limited)
--------------------

.. raw:: latex

   Fine grain (limited) parallelism augments the coarse
   grain parallelism by allowing tasks to communicate with
   the Hub prior to their conclusion. The closures
   (\Label{micro}) that would permit such communication
   must be declared and defined following the steps
   reviewed for parallel task manager closures
   (\Label{macro}).

   However,
.. raw:: latex

    in order to avoid the vast majority of
   deadlocks, the communication closures must be designed
   in a way so that all communication is initiated by the
   Spokes; the Hub must be restricted to processing
   incoming messages from the Spokes, and, if appropriate,
   replying to them.\vspace*{-10pt}

Fine grain (general)
--------------------

.. raw:: latex

   Fine grain (general) parallelism augments the fine grain
   (limited) parallelism by permitting communication among
   Spokes.

   However,
.. raw:: latex

    in order to avoid the vast majority of deadlocks, the fine grain
   (general) task manager closures must treat all Spokes under their
   control as a single unit; the premature termination of any Spoke
   must be treated as a premature termination of all Spokes.

.. raw:: latex

   \clearpage\newpage

.. #############################################################################################################
.. #############################################################################################################

.. raw:: latex

   \begin{center}
   \vspace*{1.5cm}
   \end{center}

.. figure:: fig-ppcoarsegrainExcPy.pdf
   :width: 3.5in
   :class: align-center
   :figclass: hc

   The architectural and runtime perspectives of coarse grain parallel exceptions. :label:`fig-ppcoarsegrainExcPy`

.. figure:: fig-ppfinegrainAExc.pdf
   :width: 3.5in
   :class: align-center
   :figclass: hc

   The architectural and runtime perspectives of fine grain (limited) parallel exceptions. :label:`fig-ppfinegrainAExc`

.. figure:: fig-ppfinegrainBExc.pdf
   :width: 3.5in
   :class: align-center
   :figclass: hc

   The architectural and runtime perspectives of fine grain (general) parallel exceptions. :label:`fig-ppfinegrainBExc`

.. raw:: latex

   \begin{center}
   \vspace*{1cm}
   \end{center}

Types of Fault-Tolerant Parallel Exceptions
-------------------------------------------

.. raw:: latex

    To quote Wikipedia, "exception handling is a construct designed to
    handle the occurrence of exceptions, special conditions that change the
    normal flow of program execution".

    The ability to throw and catch exceptions forms the bedrock of the
    serial Python language. We will review details of how we extended the
    basic serial exception infrastructure to account for exceptions that
    may occur across many compute resources.

    Our solution is predicated on the notion that parallel task managers
    must take ownership of how serial exceptions are handled across all
    resources under their control. Therefore, unlike in the serial world,
    the parallel exception handling infrastructure must be customized for
    each type of parallel task manager.\vspace*{-10pt}

Coarse grain
------------

.. raw:: latex

   Exception handling, as traditionally defined in the serial context,
   is designed to handle the change in the normal flow of program execution ...
   a rather straightforward concept given that there is only one call-stack.

   However,

.. raw:: latex

    when exploiting parallelism, the normal flow of program execution
   involves multiple resources and, therefore, multiple call-stacks need
   to be processed in a fault-tolerant manner. Furthermore, in order to enforce
   various forms of parallel invariants, we need an ability to throw exceptions
   at any resource, but which may only be caught by the Hub.

   Stated 

.. raw:: latex

    another way, in order to make our problem
   tractable in the context of coarse grain parallelism:
   \begin{itemize}
   \item[$\bullet$\hspace*{0.1cm}] on a Spoke, any
   uncaught/uncatchable exception must be treated and
   reported as final status of the task. Therefore, an
   exception free execution on the Hub would result in the
   normal unrolling of the call-stack at the Hub, as
   depicted by (\Label{A}, \Label{B}).
   \item[$\bullet$\hspace*{0.1cm}] on the Hub, any uncaught exception from any
   callbacks invoked by the task manager must result in
   the forcible termination and, if appropriate,
   relaunching of Spokes, as depicted by (\Label{C}, \Label{D}).
   \end{itemize}\vspace*{-10pt}

Fine grain (limited)
--------------------

.. raw:: latex

   The exception handling infrastructure in the context of fine grain (limited)
   parallelism may be identical to that for coarse grain parallelism
   provided stale replies generated by the Hub and meant for some Spoke
   can be filtered out at the Hub itself.\vspace*{-10pt}

Fine grain (general)
--------------------

.. raw:: latex

   Given that fine grain task manager closures treat all Spokes as a single unit:
   \begin{itemize}
   \item[$\bullet$\hspace*{0.1cm}] on a Spoke, any uncaught/uncatchable exception
   must be treated and reported as final status of all the
   Spokes. Therefore, an exception free execution on the Hub
   and all Spokes would result in the normal
   unrolling of the call-stack at the Hub, as depicted by
   (\Label{A}, \Label{B}).
   \item[$\bullet$\hspace*{0.1cm}] any uncaught/uncatchable exception from any
   callbacks invoked by the task manager or by any Spoke should result in
   the forcible termination and, if appropriate,
   relaunching of Spokes, as depicted by (\Label{C}, \Label{D}).\vspace*{-0.0cm}\\
   \end{itemize}

.. raw:: latex

   \clearpage\newpage

.. #############################################################################################################
.. #############################################################################################################

.. raw:: latex

   \begin{center}
   \vspace*{1cm}
   \end{center}

.. figure:: fig-pmpStack.pdf
   :width: 3.5in
   :figclass: bht

   Partition/List Parallel Management Pattern. :label:`fig-pmpStack`

.. raw:: latex

   \begin{center}
   \begin{figure}[ht]
   \noindent{\hfill\includegraphics[width=3.5in]{fig-ppcoarsegrainAppLandscape.pdf}\hfill}
   \caption{The architectural and runtime perspective of launching stand-alone applications
   in parallel using SPM.Python. \DUrole{label}{fig-ppcoarsegrainAppLandscape}}
   \end{figure}
   \vspace*{2cm}
   \end{center}

Problem Decomposition
---------------------

.. raw:: latex

   Understanding the nature of any parallel problem is key to
   determining the appropriate solution. Parallel Management Patterns
   (PMPs) provide a framework for decomposing and authoring scalable,
   fault-tolerant parallel solutions. In other-words, if the end goal
   is some parallel application, PMPs enable us to classify the journey
   to the end goal in terms of the nature of parallelism to be exploited, while
   parallel closures provided by SPM.Python enable us to express the
   parallelism implied by any PMP.

   For the purpose of illustration, we shall review  an implementation of
   the Partition/List PMP, a pattern that captures the essence of how to execute a list
   of tasks across many compute resources in a 
   fault-tolerant manner.\vspace*{-10pt}

.. raw:: latex

   \subsection*{Problem Statement}
   Our goal is to invoke the SPM coprocess API:\vspace*{-4pt}\\
   \snippet{snippet-problem-stmt.pdf}
   across multiple resources. We shall capture the context
   - in the form of arguments needed, and the final result to be returned - 
   of each execution by way of tasks. To that end, we shall augment
   the aforementioned serial functionality by authoring
   a scalable, parallel, fault-tolerant Python script made up of the
   following components:
   \begin{itemize}
   \item declaration of a (task manager) closure at the Hub,
   \item definition of tasks, processing of status reports, and
         invocation of task manager at the Hub.
   \end{itemize}
   As an aside, note that the backend of our closure
   will evaluate the task on our behalf ... a process
   that is rather straightforward given that we would be
   invoking a built-in method (\SPMtok{ShellPolicyA}).\vspace*{-10pt}

   \subsection*{\Label{A} Task manager: Declaration and Definition}

   \noindent
   In order to create (declare and define) an instance of the task manager,
   we require the Hub to be offline to in order to avoid
   various types of parallel race conditions. This invariant
   is captured by the decorator statements on lines 1 and 2.

   A natural point in time to perform this initialization step
   would be when loading the module containing the statements
   prior to actual usage. In other words, initialization should
   occur when the file containing \SPMtokB{Init} method is imported
   by the Python interpreter.

   The arguments for creating our instance bear highlighting.
   Each instance of any closure must be unique within a module;
   hence, the unique string as argument 1. Furthermore, all
   instances of our closure are defined in terms of two stages. 
   Of these, functionality for stage 1 is expected via a callback;
   hence argument 2 (\SPMtokB{TaskStat}).
.. raw:: latex

   \snippetB{snippet-demo-A.pdf}\vspace*{-14pt}
   \clearpage\newpage

.. #############################################################################################################
.. #############################################################################################################

.. raw:: latex

   \begin{center}
   \vspace*{3.5cm}
   \begin{figure}[ht]
   \noindent{\hfill\includegraphics[width=3.0in]{snippet-task-typedef.pdf}\hfill}
   \caption{Typedef for the definition of list of tasks. \DUrole{label}{snippet-task-typedef}}
   \end{figure}
   \vspace*{2.0cm}
   \begin{figure}[ht]
   \noindent{\hfill\includegraphics[scale=0.8]{snippet-exceptions.pdf}\hfill}
   \caption{Hierarchy of (parallel) SPM exceptions. \DUrole{label}{snippet-exceptions}}
   \end{figure}
   \vspace*{5cm}
   \end{center}

.. raw:: latex

   \subsection*{\Label{A} Task manager: Population and Invocation}

   \noindent
   Our goal in the function \SPMtokB{Main} is to be able to invoke the task
   manager (line 18). However, before doing  so, we must populate it 
   with the tasks to be executed. This is achieved by submitting our tasks
   by way of the API \SPMtok{Stage0}, as shown in lines 11 through 16.

   Once our task manager is invoked, the Hub transitions to the online
   state. The transition back to offline does not occur until just prior to the conclusion
   of the invocation.

.. raw:: latex
   
   \snippetB{snippet-demo-B.pdf}

.. raw:: latex

   \vspace*{-20pt}
   \subsection*{\Label{B} Task manager: (Final) Status Reports}

   \noindent
   The method \SPMtokB{TaskStat} (used when declaring and defining our
   closure) is automatically invoked by the task manager to process
   the status report of any task.  Note that this method is invoked
   while the Hub is in the online state. This invariant is captured by
   the decorator statements on lines 1 and 2.

.. raw:: latex

   \snippetB{snippet-demo-C.pdf}

.. raw:: latex

   \vspace*{-20pt}
   \subsection*{\Label{C} Task manager: Preloading of Python modules}
   \subsection*{\Label{D} Task manager: Task Evaluation}
   
   \noindent
   As each task involves the invocation of one of the built-in
   spm coprocess methods, we do not need to define any method
   to accept and evaluate any task. Instead, our task manager
   will automatically evaluate our tasks on the Spokes, and return the 
   respective status reports to the Hub.\\

.. raw:: latex

   \clearpage\newpage

.. #############################################################################################################
.. #############################################################################################################

.. raw:: latex

   \begin{center}
   \vspace*{3.05cm}
   \begin{figure}[ht]
   \noindent{\hfill\includegraphics[width=3.5in]{fig-session.pdf}\hfill}
   \caption{A typical parallel session of SPM.Python. \DUrole{label}{fig-session}}
   \end{figure}
   \vspace*{7.25cm}
   \end{center}
   \noindent
   The automatic evaluation of our tasks is aided by the typedef used
   when initializing \SPMtok{Stage0} (at the Hub). Specifically, all
   Spokes end up executing the pseudo-code:

.. raw:: latex

   \snippetC{snippet-demo-D.pdf}\vspace*{-10pt}

SPM.Python Session
------------------

.. raw:: latex

   Having reviewed our parallel application, we will conclude
   by describing an actual SPM.Python session. We start off
   by importing the \SPMtok{Pool} module (\Label{Bullet}). Next we import our
   parallel application \SPMtokB{Demo}, and run our application
   four times before exiting, as illustrated by \Label{Intra} and
   \Label{Inter}.

   The 
.. raw:: latex

    first two times (marked \Label{Intra}), we limited ourselves to
   cores from the server running the Hub. \SPMtokB{IntraOnePerServer}
   refers to one unique core on the server.

   The
.. raw:: latex

    second two times (marked \Label{Inter}), we limited our selves to
   cores from potentially different servers. \SPMtokB{InterOnePerServer}
   refers to one unique core from each server.

   The
.. raw:: latex

    fact that the results produced are identical should not be a
   surprise since our code is a function of a handle to a pool, and
   not its content. In other words, user code remains unchanged
   despite having selected four different sets of resources.

   Note that, notwithstanding our rather small script, our solution
   is not only fault-tolerant (thanks to closures), self-cleaning
   (thanks to robust timeout support), but also robust
   (thanks to the efficient manner by which parallel invariants
   are enforced). So, once we have tested our solution in a serial-like
   environment, we can be sure our solution can be deployed on any cluster.
   See \cite{PMP02} for a comprehensive list of problem decomposition using other 
   PMPs including self contained and equally powerful examples.\vspace*{-10pt}

Conclusion
----------

.. raw:: latex

   In this paper, we reviewed the technical anatomy of SPM.Python, a
   scalable parallel version of the serial Python language.  We began
   with a prologue presenting the acid test for general purpose
   parallel computing. Next, we described the solution to four
   formerly open technical problems, namely the decoupling of tracking
   of resources from management of resources; the declaration and
   definition of parallel closures; the design and architecture of
   parallel closures that delineate serial and parallel components;
   and fault-tolerant parallel exception handling. We concluded by
   illustrating how a parallel problem, once classified in terms of a
   Parallel Management Pattern (PMP), can be decomposed and easily
   expressed in terms of SPM.Python's parallel closures.\vspace*{-10pt}

.. raw:: latex

   \begin{thebibliography}{01}
   \bibitem[1]{Celery}{
   Celery, {\href{http://celeryproject.org}{celeryproject.org}}}
   \bibitem[2]{PPy}{
   Parallel Python, {\href{http://www.parallelpython.com}{www.parallelpython.com}}}
   \bibitem[3]{Disco}{
   Disco, {\href{http://discoproject.org}{discoproject.org}}}
   \bibitem[4]{PaPy}{
   PaPy, {\href{http://code.google.com/p/papy}{code.google.com/p/papy}}}
   \bibitem[5]{MPI}{
   PyMPI, {\href{http://mpi4py.sourceforge.net}{mpi4py.sourceforge.net}}}
   \bibitem[6]{PVM}{
   PyPVM, {\href{http://pypvm.sourceforge.net}{pypvm.sourceforge.net}}}
   \bibitem[7]{USP01}{
   Minesh B. Amin,. \emph{Resource Tracking Method and Apparatus},
   \href{http://www.mbasciences.com/Patents.html}{United States Patent \#: 7,926,058 B2, April 12, 2011.}}
   \bibitem[8]{PMP02}{
   Parallel Management Patterns, {\href{http://www.mbasciences.com/pmp.html}{www.mbasciences.com/pmp.html}}}
   \end{thebibliography}
   \clearpage\newpage

.. #############################################################################################################
.. #############################################################################################################

Appendix A
----------

.. raw:: latex
   
   Figures 14 through 16 highlight the manner by which any module can
   be parallelized using SPM.Python.  Specifically, a serial module
   that computes number of prime numbers within a given range (Figure
   14) is parallelized by introducing two wrappers as depicted by
   Figure 15 (for Spoke), and Figure 16 (for Hub). Recall that
   SPM.Python has built-in support for multiple different and distinct
   forms of parallelism. However, for our purpose, we are only
   interested in the closure that executes a list of tasks in
   parallel.

   \begin{figure}[hbt]
   \includegraphics[width=3.1in]{fig-mainSpokeSerial.pdf}
   \caption{Spoke: Original 'serial' module that computes the number of
   prime numbers given a range.}
   \vspace*{30pt}
   \includegraphics[width=3.1in]{fig-mainSpoke.pdf}
   \caption{Spoke: Wrapper around serial functionality. The wrapper is
   automatically invoked by SPM.Python based on the content of the task's 'spm'
   sub-structure.}
   \end{figure}

   \begin{figure}[hbt]
   \includegraphics[width=3.4in]{fig-mainHub.pdf}
   \caption{Hub: Creation/population/invocation of parallel (task manager) closure.
   The backend of the closure, once invoked, would execute as many tasks in parallel
   as possible using resources within the {\bf pool}.}
   \end{figure}


